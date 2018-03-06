/*
 * cv_utils.cpp
 *
 *  Created on: Dec 7, 2012
 *  	Author: tan
 *  Related blog post at:
 *	http://sidekick.windforwings.com/2012/12/opencv-separating-items-on-store-shelves.html
 *
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <stdlib.h>

#include "cv_utils.h"

/*****************************************************
 * print what optimization libraries are available
 *****************************************************/
void print_lib_version() {
	const char* libraries;
	const char* modules;
	cvGetModuleInfo(NULL, &libraries, &modules);
	printf("Libraries: %s\nModules: %s\n", libraries, modules);
}

/*****************************************************
 * comparison functions used for sorting lines and points
 *****************************************************/
int compare_line_y_then_x(const void *_l1, const void *_l2, void* userdata) {
	CvPoint* l1 = (CvPoint*)_l1;
	CvPoint* l2 = (CvPoint*)_l2;
	int y_diff = int(l1[0].y - l2[0].y);
	if(0 != y_diff) return y_diff;

	int x_diff = int(l1[0].x - l2[0].x);
	return x_diff;
}

int compare_line_x_then_y(const void *_l1, const void *_l2, void* userdata) {
	CvPoint* l1 = (CvPoint*)_l1;
	CvPoint* l2 = (CvPoint*)_l2;
	int x_diff = int(l1[0].x - l2[0].x);
	if(0 != x_diff) return x_diff;

	int y_diff = int(l1[0].y - l2[0].y);

	return y_diff;
}

int compare_int(const void *i1, const void *i2) {
	int diff = *((int *)i1) - *((int *)i2);
	return diff;
}


/*****************************************************
 * convert lines from rho-theta to x-y space
 *****************************************************/
CvPoint * rho_theta_to_line(float *rho_theta, CvPoint *line, int max_dim) {
	float rho = rho_theta[0];
	float theta = rho_theta[1];

	double a = cos(theta), b = sin(theta);
	double x0 = a * rho, y0 = b * rho;
	line[0].x = cvRound(x0 + max_dim * (-b));
	line[0].y = cvRound(y0 + max_dim * (a));
	line[1].x = cvRound(x0 - max_dim * (-b));
	line[1].y = cvRound(y0 - max_dim * (a));

	if(line[0].x < 0) line[0].x = 0;
	if(line[0].y < 0) line[0].y = 0;
	if(line[1].x < 0) line[1].x = 0;
	if(line[1].y < 0) line[1].y = 0;

	return line;
}

/*****************************************************
 * from a sequence of lines, select only lines
 * that match certain conditions
 *****************************************************/
CvSeq* select_lines_by_separation(CvSeq *ret, int sep, float accuracy, bool is_y) {
	CvPoint* line1 = (CvPoint*)cvGetSeqElem(ret, 0);
	for(int line_idx = 1; line_idx < ret->total; line_idx++) {
		CvPoint* line2 = (CvPoint*)cvGetSeqElem(ret, line_idx);
		int this_dist = is_y ? abs(line1[0].y - line2[0].y) : abs(line1[0].x - line2[0].x);

		if(this_dist < (accuracy*sep)) {
			//printf("Ignoring line %d. Less than sep [%d]\n", line_idx, this_dist);
			cvSeqRemove(ret, line_idx);
			line_idx--;
		}
		else {
			//printf("Keeping line %d. More than sep [%d]\n", line_idx, this_dist);
			line1 = line2;
		}
	}
	return ret;
}

CvSeq* select_lines_with_angle(CvSeq *ret, double angle, double accuracy) {
	assert(angle < CV_PI);
	assert(accuracy < (CV_PI/8));

	for(int line_idx = 0; line_idx < ret->total; line_idx++) {
		CvPoint* line = (CvPoint*)cvGetSeqElem(ret, line_idx);
		double y_dist = fabs(line[1].y - line[0].y) + 1;
		double x_dist = fabs(line[1].x - line[0].x) + 1;
		double theta = atan(y_dist/x_dist);

		// consider only from 0 to PI
		if(theta >= CV_PI) theta -= CV_PI;

		if(fabs(theta - angle) > accuracy) {
			/*
			printf("removing line %d,%d->%d,%d with angle [%f] [%f] angle[%f] accuracy[%f]\n",
					line[0].x, line[0].y, line[1].x, line[1].y,
					theta, theta * 180 / CV_PI, angle, accuracy);
			*/
			cvSeqRemove(ret, line_idx);
			line_idx--;
		}
		/*else {
			printf("keeping line %d,%d->%d,%d with angle [%f] [%f] angle[%f] accuracy[%f]\n",
					line[0].x, line[0].y, line[1].x, line[1].y,
					theta, theta * 180 / CV_PI, angle, accuracy);
		}*/
	}
	return ret;
}

int get_median_dist(int *dist_arr, int dist_arr_cnt) {
	if(0 == dist_arr_cnt) return 0;

	// sort the distances
	qsort(dist_arr, dist_arr_cnt, sizeof(int), compare_int);
	/*
	printf("Sorted distances: ");
	for(int line_idx=0; line_idx < dist_arr_cnt; line_idx++) printf("%d, ", dist_arr[line_idx]);
	printf("\n");
	*/

	return dist_arr[(dist_arr_cnt-1)/2];
}

double avg_line_dist(CvSeq *lines, bool is_y) {
	double dist = 0;
	CvPoint* line1 = (CvPoint*)cvGetSeqElem(lines, 0);
	for(int line_idx = 1; line_idx < lines->total; line_idx++) {
		CvPoint* line2 = (CvPoint*)cvGetSeqElem(lines, line_idx);
		if(is_y) dist += abs(line1[0].y - line2[0].y);
		else dist += abs(line1[0].x - line2[0].x);
		line1 = line2;
	}
	return (dist / lines->total);
}

void print_lines(CvSeq *lines) {
	for(int line_idx = 0; line_idx < lines->total; line_idx++) {
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, line_idx);
		printf("line %d,%d->%d,%d\n", line[0].x, line[0].y, line[1].x, line[1].y);
	}
}

void draw_lines(CvSeq* lines, IplImage *img_color_dst, int col_idx) {
	static CvScalar colors[LINE_DRAW_NUM_COLORS] = { CV_RGB(255,0,0), CV_RGB(0,255,0), CV_RGB(0,0,255) };

	for(int line_idx = 0; line_idx < lines->total; line_idx++) {
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines, line_idx);
		cvLine(img_color_dst, line[0], line[1], colors[col_idx], 2, CV_AA);
	}
}

CvSeq * find_hough_lines(IplImage *img, CvMemStorage *mem_store, int hough_method, double rho, double theta, int threshold, double p1, double p2, int max_dim) {
	CvSeq *lines = cvHoughLines2(img, mem_store, hough_method, rho, theta, threshold, p1, p2);

	CvSeq *ret = NULL;
	if((0 == hough_method) || (1 == hough_method)) {
		ret = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint)*2, mem_store);
		for(int line_idx = 0; line_idx < lines->total; line_idx++) {
			CvPoint line_ends[2];
			cvSeqPush(ret, rho_theta_to_line((float *)cvGetSeqElem(lines, line_idx), line_ends, max_dim));
		}
	}
	else {
		ret = lines;
	}

	return ret;
}

/*****************************************************
 * smoothen the image and adjust brightness and contrast
 *****************************************************/
void smooth_brightness_contrast(IplImage *img, IplImage *img_result, int brightness, int contrast) {
	cvSmooth(img, img_result, CV_BILATERAL, 5, 5, 30, 30);

	// increase contrast and adjust brightness
	cvAddWeighted(img_result, 0.5, img_result, 0.5, brightness, img_result);

	// increase contrast further if specified
	for(int contrast_idx = 0; contrast_idx < contrast; contrast_idx++) {
		cvAddWeighted(img_result, 1, img_result, 1, 0, img_result);
	}
}
