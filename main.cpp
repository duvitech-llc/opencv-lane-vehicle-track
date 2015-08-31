/*
Notes: 
	1) keep tracing limited, console io is costly
	2) if using camera as input make sure to shutdown correctly
 */
#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <math.h>
#include "utils.h"

#undef MIN
#undef MAX
#define MAX(a,b) ((a)<(b)?(b):(a))
#define MIN(a,b) ((a)>(b)?(b):(a))
#define TRACE(...) fprintf(stdout, __VA_ARGS__)

using namespace cv;

void imageCrop(IplImage* src,  IplImage* dest, CvRect rect) {
    cvSetImageROI(src, rect); 
    cvCopy(src, dest); 
    cvResetImageROI(src); 
}

struct Lane {
	Lane(){}
	Lane(CvPoint a, CvPoint b, float angle, float kl, float bl): p0(a),p1(b),angle(angle),
		votes(0),visited(false),found(false),k(kl),b(bl) { }

	CvPoint p0, p1;
	int votes;
	bool visited, found;
	float angle, k, b;
};

struct Status {
	Status():reset(true),lost(0){}
	ExpMovingAverage k, b;
	bool reset;
	int lost;
	int side;
};

struct Vehicle {
	CvPoint bmin, bmax;
	int symmetryX;
	bool valid;
	unsigned int lastUpdate;
};

struct VehicleSample {
	CvPoint center;
	float radi;
	unsigned int frameDetected;
	int vehicleIndex;
};

#define GREEN CV_RGB(0,255,0)
#define RED CV_RGB(255,0,0)
#define BLUE CV_RGB(255,0,255)
#define PURPLE CV_RGB(255,0,255)
#define LANES_RGB CV_RGB(255, 0, 255)
#define LANES_BLUE CV_RGB(0, 0, 255)
#define LANES_RED CV_RGB(255, 0, 0)

Status g_laneR, g_laneL;
std::vector<Vehicle> g_vehiclesList;
std::vector<VehicleSample> g_samplesList;

/*
enum{
	SCAN_STEP = 5,			  // in pixels ORIG
	LINE_REJECT_DEGREES = 10, // in degrees ORIG
	BW_TRESHOLD = 250,		  // edge response strength to recognize for 'WHITE' ORIG
	BORDERX = 10,			  // px, skip this much from left & right borders
	MAX_RESPONSE_DIST = 5,	  // px

	CANNY_MIN_TRESHOLD = 1,	  // edge detector minimum hysteresis threshold
	CANNY_MAX_TRESHOLD = 100, // edge detector maximum hysteresis threshold

	HOUGH_TRESHOLD = 50,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 50,	// remove lines shorter than this treshold ORIG
	HOUGH_MAX_LINE_GAP = 100,   // join lines to one with smaller than this gaps ORIG

	CAR_DETECT_LINES = 4,    // minimum lines for a region to pass validation as a 'CAR'
	CAR_H_LINE_LENGTH = 10,  // minimum horizontal line length from car body in px

	MAX_VEHICLE_SAMPLES = 30,      // max vehicle detection sampling history
	CAR_DETECT_POSITIVE_SAMPLES = MAX_VEHICLE_SAMPLES-2, // probability positive matches for valid car
	MAX_VEHICLE_NO_UPDATE_FREQ = 15 // remove car after this much no update frames
};
*/
enum{
	SCAN_STEP = 2,			  // in pixels
	LINE_REJECT_DEGREES = 24, // in degrees
	BW_TRESHOLD = 253,		  // edge response strength to recognize for 'WHITE'
	BORDERX = 10,			  // px, skip this much from left & right borders
	MAX_RESPONSE_DIST = 5,	  // px

	CANNY_MIN_TRESHOLD = 75,	  // edge detector minimum hysteresis threshold
	CANNY_MAX_TRESHOLD = 95, // edge detector maximum hysteresis threshold

	HOUGH_TRESHOLD = 50,		// line approval vote threshold
	HOUGH_MIN_LINE_LENGTH = 30,	// remove lines shorter than this treshold
	HOUGH_MAX_LINE_GAP = 80,   // join lines to one with smaller than this gaps

	CAR_DETECT_LINES = 6,    // minimum lines for a region to pass validation as a 'CAR'
	CAR_H_LINE_LENGTH = 6,  // minimum horizontal line length from car body in px

	MAX_VEHICLE_SAMPLES = 15,      // max vehicle detection sampling history
	CAR_DETECT_POSITIVE_SAMPLES = MAX_VEHICLE_SAMPLES-2, // probability positive matches for valid car
	MAX_VEHICLE_NO_UPDATE_FREQ = 19 // remove car after this much no update frames
};
/*
#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30
*/
#define K_VARY_FACTOR 0.25f
#define B_VARY_FACTOR 18
#define MAX_LOST_FRAMES 18

void findResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list)
{
	// scans for single response: /^\_

	const int row = y * img->width * img->nChannels;
	unsigned char* ptr = (unsigned char*)img->imageData;

	int step = (endX < startX) ? -1: 1;
	int range = (endX > startX) ? endX-startX+1 : startX-endX+1;

	for(int x = startX; range>0; x += step, range--)
	{
		if(ptr[row + x] <= BW_TRESHOLD) continue; // skip black: loop until white pixels show up

		// first response found
		int idx = x + step;

		// skip same response(white) pixels
		while(range > 0 && ptr[row+idx] > BW_TRESHOLD) {
			idx += step;
			range--;
		}

		// reached black again
		if(ptr[row+idx] <= BW_TRESHOLD) {
			list.push_back(x);
		}

		x = idx; // begin from new pos
	}
}

unsigned char pixel(IplImage* img, int x, int y) {
	return (unsigned char)img->imageData[(y*img->width+x)*img->nChannels];
}

int findSymmetryAxisX(IplImage* half_frame, CvPoint bmin, CvPoint bmax) {
  
  float value = 0;
  int axisX = -1; // not found
  
  int xmin = bmin.x;
  int ymin = bmin.y;
  int xmax = bmax.x;
  int ymax = bmax.y;
  int half_width = half_frame->width/2;
  int maxi = 1;

  for(int x=xmin, j=0; x<xmax; x++, j++) {
	float HS = 0;
	for(int y=ymin; y<ymax; y++) {
		int row = y*half_frame->width*half_frame->nChannels;
		for(int step=1; step<half_width; step++) {
		  int neg = x-step;
		  int pos = x+step;
		  unsigned char Gneg = (neg < xmin) ? 0 : (unsigned char)half_frame->imageData[row+neg*half_frame->nChannels];
		  unsigned char Gpos = (pos >= xmax) ? 0 : (unsigned char)half_frame->imageData[row+pos*half_frame->nChannels];
		  HS += abs(Gneg-Gpos);
		}
	}

	if (axisX == -1 || value > HS) { // find minimum
		axisX = x;
		value = HS;
	}
  }

  return axisX;
}

bool hasVertResponse(IplImage* edges, int x, int y, int ymin, int ymax) {
	bool has = (pixel(edges, x, y) > BW_TRESHOLD);
	if (y-1 >= ymin) 
		has &= (pixel(edges, x, y-1) < BW_TRESHOLD);
	if (y+1 < ymax) 
		has &= (pixel(edges, x, y+1) < BW_TRESHOLD);
	return has;
}

int horizLine(IplImage* edges, int x, int y, CvPoint bmin, CvPoint bmax, int maxHorzGap) {

	// scan to right
	int right = 0;
	int gap = maxHorzGap;
	for (int xx=x; xx<bmax.x; xx++) {
		if (hasVertResponse(edges, xx, y, bmin.y, bmax.y)) {
			right++;
			gap = maxHorzGap; // reset
		} else {
			gap--;
			if (gap <= 0) {
				break;
			}
		}
	}

	int left = 0;
	gap = maxHorzGap;
	for (int xx=x-1; xx>=bmin.x; xx--) {
		if (hasVertResponse(edges, xx, y, bmin.y, bmax.y)) {
			left++;
			gap = maxHorzGap; // reset
		} else {
			gap--;
			if (gap <= 0) {
				break;
			}
		}
	}

	return left+right;
}

bool vehicleValid(IplImage* half_frame, IplImage* edges, Vehicle* v, int& index) {

	index = -1;

	// first step: find horizontal symmetry axis
	v->symmetryX = findSymmetryAxisX(half_frame, v->bmin, v->bmax);
	if (v->symmetryX == -1) return false;

	// second step: cars tend to have a lot of horizontal lines
	int hlines = 0;
	for (int y = v->bmin.y; y < v->bmax.y; y++) {		
		if (horizLine(edges, v->symmetryX, y, v->bmin, v->bmax, 2) > CAR_H_LINE_LENGTH) {
#if _DEBUG
			cvCircle(half_frame, cvPoint(v->symmetryX, y), 2, PURPLE);
#endif
			hlines++;
		}
	}

	int midy = (v->bmax.y + v->bmin.y)/2;

	// third step: check with previous detected samples if car already exists
	int numClose = 0;
	float closestDist = 0;
	for (int i = 0; i < g_samplesList.size(); i++) {
		int dx = g_samplesList[i].center.x - v->symmetryX;
		int dy = g_samplesList[i].center.y - midy;
		float Rsqr = dx*dx + dy*dy;
		
		if (Rsqr <= g_samplesList[i].radi*g_samplesList[i].radi) {
			numClose++;
			if (index == -1 || Rsqr < closestDist) {
				index = g_samplesList[i].vehicleIndex;
				closestDist = Rsqr;
			}
		}
	}

	return (hlines >= CAR_DETECT_LINES || numClose >= CAR_DETECT_POSITIVE_SAMPLES);
}

void removeOldVehicleSamples(unsigned int currentFrame) {
	// statistical sampling - clear very old samples
	std::vector<VehicleSample> sampl;
	for (int i = 0; i < g_samplesList.size(); i++) {
		if (currentFrame - g_samplesList[i].frameDetected < MAX_VEHICLE_SAMPLES) {
			sampl.push_back(g_samplesList[i]);
		}
	}
	g_samplesList = sampl;
}

void removeSamplesByIndex(int index) {
	// statistical sampling - clear very old samples
	std::vector<VehicleSample> sampl;
	for (int i = 0; i < g_samplesList.size(); i++) {
		if (g_samplesList[i].vehicleIndex != index) {
			sampl.push_back(g_samplesList[i]);
		}
	}
	g_samplesList = sampl;
}

void removeLostVehicles(unsigned int currentFrame) {
	// remove old unknown/false vehicles & their samples, if any
	for (int i=0; i<g_vehiclesList.size(); i++) {
		if (g_vehiclesList[i].valid && currentFrame - g_vehiclesList[i].lastUpdate >= MAX_VEHICLE_NO_UPDATE_FREQ) {
			//TRACE("\tremoving inactive car, index = %d\n", i);
			removeSamplesByIndex(i);
			g_vehiclesList[i].valid = false;
		}
	}
}

void vehicleDetection(IplImage* half_frame, CvHaarClassifierCascade* cascade, CvMemStorage* haarStorage) {

	static unsigned int frame = 0;
	static unsigned int last_cars = 0;
	frame++;
	//TRACE("*** vehicle detector frame: %d ***\n", frame);

	removeOldVehicleSamples(frame);

	// Haar Car detection
	const double scale_factor = 1.05; // every iteration increases scan window by 5%
	const int min_neighbours = 2; // minus 1, number of rectangles, that the object consists of
	CvSeq* rects = cvHaarDetectObjects(half_frame, cascade, haarStorage, 
			scale_factor, min_neighbours, CV_HAAR_DO_CANNY_PRUNING);

	// Canny edge detection of the minimized frame
	if (rects->total > 0) {
		if(last_cars != rects->total) {
			//TRACE("\thaar detected %d car hypotheses\n", rects->total);
			last_cars = rects->total;
		}
		IplImage *edges = cvCreateImage(cvSize(half_frame->width, half_frame->height), IPL_DEPTH_8U, 1);
		cvCanny(half_frame, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

		// validate vehicles 
		for (int i = 0; i < rects->total; i++) {
			CvRect* rc = (CvRect*)cvGetSeqElem(rects, i);
			
			Vehicle v;
			v.bmin = cvPoint(rc->x, rc->y);
			v.bmax = cvPoint(rc->x + rc->width, rc->y + rc->height);
			v.valid = true;

			int index;
			if (vehicleValid(half_frame, edges, &v, index)) { // put a sample on that position
				
				if (index == -1) { // new car detected

					v.lastUpdate = frame;

					// re-use already created but inactive vehicles
					for(int j=0; j<g_vehiclesList.size(); j++) {
						if (g_vehiclesList[j].valid == false) {
							index = j;
							break;
						}
					}
					if (index == -1) { // all space used
						index = g_vehiclesList.size();
						g_vehiclesList.push_back(v);
					}
					TRACE("\tnew car detected, index = %d\n", index);
				} else {
					// update the position from new data
					g_vehiclesList[index] = v;
					g_vehiclesList[index].lastUpdate = frame;
					//TRACE("\tcar updated, index = %d\n", index);
				}

				VehicleSample vs;
				vs.frameDetected = frame;
				vs.vehicleIndex = index;
				vs.radi = (MAX(rc->width, rc->height))/4; // radius twice smaller - prevent false positives
				vs.center = cvPoint((v.bmin.x+v.bmax.x)/2, (v.bmin.y+v.bmax.y)/2);
				g_samplesList.push_back(vs);
			}
		}

		//cvShowImage("Half-frame[edges]", edges);
		//cvMoveWindow("Half-frame[edges]", half_frame->width*2, half_frame->height+15); 
		cvReleaseImage(&edges);
	} else {
		//TRACE("\tno vehicles detected in current frame!\n");
	}

	removeLostVehicles(frame);

	//TRACE("\ttotal vehicles on screen: %d\n", g_vehiclesList.size());
}

void drawVehicles(IplImage* half_frame) {

	// show vehicles
	for (int i = 0; i < g_vehiclesList.size(); i++) {
		Vehicle* v = &g_vehiclesList[i];
		if (v->valid) {
			cvRectangle(half_frame, v->bmin, v->bmax, GREEN, 1);
			
			int midY = (v->bmin.y + v->bmax.y) / 2;
			cvLine(half_frame, cvPoint(v->symmetryX, midY-10), cvPoint(v->symmetryX, midY+10), PURPLE);
		}
	}

	// show vehicle position sampling
	/*for (int i = 0; i < g_samplesList.size(); i++) {
		cvCircle(half_frame, cvPoint(g_samplesList[i].center.x, g_samplesList[i].center.y), g_samplesList[i].radi, RED);
	}*/
}

void processSide(std::vector<Lane> lanes, IplImage *edges, bool right) {

	Status* side = right ? &g_laneR : &g_laneL;

	// response search
	int w = edges->width;
	int h = edges->height;
	const int BEGINY = 0;
	const int ENDY = h-1;
	const int ENDX = right ? (w-BORDERX) : BORDERX;
	int midx = w/2;
	int midy = edges->height/2;
	unsigned char* ptr = (unsigned char*)edges->imageData;

	// show responses
	int* votes = new int[lanes.size()];
	// clear votes
	for(int i=0; i<lanes.size(); i++) votes[i++] = 0;

	for(int y=ENDY; y>=BEGINY; y-=SCAN_STEP) {
		std::vector<int> rsp;
		findResponses(edges, midx, ENDX, y, rsp);

		if (rsp.size() > 0) {
			int response_x = rsp[0]; // use first reponse (closest to screen center)

			float dmin = 9999999;
			float xmin = 9999999;
			int match = -1;
			for (int j=0; j<lanes.size(); j++) {
				// compute response point distance to current line
				float d = dist2line(
						cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y), 
						cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y), 
						cvPoint2D32f(response_x, y));

				// point on line at current y line
				int xline = (y - lanes[j].b) / lanes[j].k;
				int dist_mid = abs(midx - xline); // distance to midpoint

				// pick the best closest match to line & to screen center
				if (match == -1 || (d <= dmin && dist_mid < xmin)) {
					dmin = d;
					match = j;
					xmin = dist_mid;
					break;
				}
			}

			// vote for each line
			if (match != -1) {
				votes[match] += 1;
			}
		}
	}

	int bestMatch = -1;
	int mini = 9999999;
	for (int i=0; i<lanes.size(); i++) {
		int xline = (midy - lanes[i].b) / lanes[i].k;
		int dist = abs(midx - xline); // distance to midpoint

		if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
			bestMatch = i;
			mini = dist;
		}
	}

	if (bestMatch != -1) {
		Lane* best = &lanes[bestMatch];
		float k_diff = fabs(best->k - side->k.get());
		float b_diff = fabs(best->b - side->b.get());

		bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;
/*
		TRACE("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n", 
			(right?"RIGHT":"LEFT"), k_diff, b_diff, (update_ok?"no":"yes"));
*/		
		if (update_ok) {
			// update is in valid bounds
			side->k.add(best->k);
			side->b.add(best->b);
			side->reset = false;
			side->lost = 0;
		} else {
			// can't update, lanes flicker periodically, start counter for partial reset!
			side->lost++;
			if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
				side->reset = true;
			}
		}

	} else {
		TRACE("no %s lanes detected - lane tracking lost! counter increased\n", right?"RIGHT":"LEFT");
		side->lost++;
		if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
			// do full reset when lost for more than N frames
			side->reset = true;
			side->k.clear();
			side->b.clear();
		}
	}

	delete[] votes;
}

void processLanes(CvSeq* lines, IplImage* edges, IplImage* temp_frame) {

	// classify lines to left/right side
	std::vector<Lane> left, right;

	for(int i = 0; i < lines->total; i++ )
	{
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
		int dx = line[1].x - line[0].x;
		int dy = line[1].y - line[0].y;
		float angle = atan2f(dy, dx) * 180/CV_PI;

		if (fabs(angle) <= LINE_REJECT_DEGREES) { // reject near horizontal lines
			continue;
		}

		// assume that vanishing point is close to the image horizontal center
		// calculate line parameters: y = kx + b;
		dx = (dx == 0) ? 1 : dx; // prevent DIV/0!  
		float k = dy/(float)dx;
		float b = line[0].y - k*line[0].x;

		// assign lane's side based by its midpoint position 
		int midx = (line[0].x + line[1].x) / 2;
		if (midx < temp_frame->width/2) {
			left.push_back(Lane(line[0], line[1], angle, k, b));
		} else if (midx > temp_frame->width/2) {
			right.push_back(Lane(line[0], line[1], angle, k, b));
		}
	}
/*
	// show Hough lines
	for	(int i=0; i<right.size(); i++) {
		cvLine(temp_frame, right[i].p0, right[i].p1, LANES_BLUE, 2);
	}

	for	(int i=0; i<left.size(); i++) {
		cvLine(temp_frame, left[i].p0, left[i].p1, LANES_RED, 2);
	}
*/
	processSide(left, edges, false);
	processSide(right, edges, true);

	// show computed lanes
	int x = temp_frame->width * 0.55f;
	int x2 = temp_frame->width;
	cvLine(temp_frame, cvPoint(x, g_laneR.k.get()*x + g_laneR.b.get()), 
		cvPoint(x2, g_laneR.k.get() * x2 + g_laneR.b.get()), LANES_RGB, 4, CV_AA);

	x = temp_frame->width * 0;
	x2 = temp_frame->width * 0.45f;
	cvLine(temp_frame, cvPoint(x, g_laneL.k.get()*x + g_laneL.b.get()), 
		cvPoint(x2, g_laneL.k.get() * x2 + g_laneL.b.get()), LANES_RGB, 4, CV_AA);
}

int main(int argc, char** argv)
{

	//rtsp://184.72.239.149/vod/mp4:BigBuckBunny_115k.mov
	//ref: http://stackoverflow.com/questions/21041370/opencv-how-to-capture-rtsp-video-stream
	//ref: https://www.videolan.org/doc/streaming-howto/en/ch04.html
	// open an input source
	CvCapture *input_video = NULL;
	if(argc == 1) {
		TRACE("open avi file\n");
		input_video = cvCreateFileCapture("road.avi");
	} else {
		TRACE("open stream\n");
		input_video = cvCaptureFromCAM(0);
	}

	if (input_video == NULL) {
		TRACE( "Error: Can't open video\n");
		return -1;
	}

	TRACE("HERE: %s:%d\n", __FILE__, __LINE__);
	//CvFont font;
	//cvInitFont( &font, CV_FONT_VECTOR0, 0.25f, 0.25f);

	// get video or camera image size
	CvSize video_size;
	video_size.height = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_HEIGHT);
	video_size.width = (int) cvGetCaptureProperty(input_video, CV_CAP_PROP_FRAME_WIDTH);

	long current_frame = 0;
	int key_pressed = -1;
	IplImage *frame = NULL;

	CvSize frame_size = cvSize(video_size.width, video_size.height/2);
	IplImage *temp_frame = cvCreateImage(frame_size, IPL_DEPTH_8U, 3);
	IplImage *grey = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
	IplImage *edges = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
	IplImage *half_frame = cvCreateImage(cvSize(video_size.width/2, video_size.height/2), IPL_DEPTH_8U, 3);

	CvMemStorage* houghStorage = cvCreateMemStorage(0);
	CvMemStorage* haarStorage = cvCreateMemStorage(0);
	CvHaarClassifierCascade* cascade = (CvHaarClassifierCascade*)cvLoad("haar/cars3.xml");

	//cvSetCaptureProperty(input_video, CV_CAP_PROP_POS_FRAMES, current_frame);
	while(key_pressed < 0) {

		frame = cvQueryFrame(input_video);
		if (frame == NULL) {
			TRACE( "Error: null frame received\n");
			return -1;
		}

		cvPyrDown(frame, half_frame, CV_GAUSSIAN_5x5); // Reduce the image by 2	 
		//cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale

		// we're interested only in road below horizont - so crop top image portion off
		imageCrop(frame, temp_frame, cvRect(0,frame_size.height,frame_size.width,frame_size.height));
		cvCvtColor(temp_frame, grey, CV_BGR2GRAY); // convert to grayscale
		
		// Perform a Gaussian blur ( Convolving with 5 X 5 Gaussian) & detect edges
		//cvSmooth(grey, grey, CV_GAUSSIAN, 5, 5); // orig
		cvSmooth(grey, grey, CV_GAUSSIAN, 3, 3);
		//cvSmooth(grey, grey, CV_BLUR, 9, 9);
		cvCanny(grey, edges, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

		// do Hough transform to find lanes
		double rho = 1.1; // orig=1
		double theta = CV_PI/180;
		CvSeq* lines = cvHoughLines2(edges, houghStorage, CV_HOUGH_PROBABILISTIC, 
			rho, theta, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

		processLanes(lines, edges, temp_frame);
		
		// process vehicles
		vehicleDetection(half_frame, cascade, haarStorage);
		drawVehicles(half_frame);

		cvShowImage("Half-frame", half_frame);
		cvMoveWindow("Half-frame", half_frame->width*2-15, 0); 

		// show middle line
		cvLine(temp_frame, cvPoint(frame_size.width/2,0), 
			cvPoint(frame_size.width/2,frame_size.height), CV_RGB(255, 255, 0), 1);

		cvShowImage("Grey", grey);
		cvShowImage("Edges", edges);
		cvShowImage("Color", temp_frame);
		
		cvMoveWindow("Grey", 0, 0); 
		cvMoveWindow("Edges", 0, frame_size.height+65);
		cvMoveWindow("Color", 0, 2*(frame_size.height+65)); 

		key_pressed = cvWaitKey(15); // orig=5
	}

	//TRACE("HERE: %s:%d\n", __FILE__, __LINE__);
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&haarStorage);
	cvReleaseMemStorage(&houghStorage);

	cvReleaseImage(&grey);
	cvReleaseImage(&edges);
	cvReleaseImage(&temp_frame);
	cvReleaseImage(&half_frame);

	cvReleaseCapture(&input_video);
	//TRACE("END argc=%d\n", argc);
}
