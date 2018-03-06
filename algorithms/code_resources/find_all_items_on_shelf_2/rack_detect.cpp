/*
 * rack_detect.cpp
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

#define MAX_SHELVES 		7
#define MAX_ITEMS_PER_ROW	30

const char	*WINDOW_NAME = "Store Item Detect";
const int HOUGH_METHODS[3] = { CV_HOUGH_STANDARD, CV_HOUGH_MULTI_SCALE, CV_HOUGH_PROBABILISTIC };

int param_disp_image = 5;				// the image to be displayed in the debug window
int param_hough_method = 0;				// the hough method to use. right now fixed to CV_HOUGH_STANDARD
int param_h_rho = 0;					// pixels granularity for rho/distance (=param+1)
int param_h_theta = 71;					// angle granularity for theta (=2*pi/(param+1))
int param_h_threshold = 75;				// number of points that should support the line (used as param+1 %)
int param_v_rho = 0;					// pixels granularity for rho/distance (=param+1)
int param_v_theta = 71;					// angle granularity for theta (=2*pi/(param+1))
int param_v_threshold = 50;				// number of points that should support the line (used as param+1 %)
int param_p1 = 0;						// increase accuracy of rho by factor
int param_p2 = 1000;					// increase accuracy of theta by factor
int param_contrast_factor = 0;			// increase contrast by factor
int param_brightness_factor = 50;		// increase brightness by factor (50 is no change)

// image attributes kept as globals and used at different places
int img_attrib_depth;
int img_attrib_channels;
CvSize img_attrib_dim;

IplImage		*img_read;
IplImage		*img_src;
IplImage		*img_smooth;
IplImage		*img_edges;
IplImage 		*img_h_edge_src;
IplImage 		*img_v_edge_src;
IplImage		*img_color_dst;
CvMemStorage	*mem_store;
CvMat 			*h_smudge_kernel;
CvMat 			*v_smudge_kernel;

CvTrackbarCallback on_change CV_DEFAULT(NULL);

bool param_changed = true;

void on_param_change(int pos) {
	param_changed = true;
}

void create_gui() {
	cvNamedWindow(WINDOW_NAME);
	cvCreateTrackbar("Brightness", WINDOW_NAME, &param_brightness_factor, 100, on_param_change);
	cvCreateTrackbar("Contrast", WINDOW_NAME, &param_contrast_factor, 10, on_param_change);

	cvCreateTrackbar("Display Image", WINDOW_NAME, &param_disp_image, 5, on_param_change);

	// threshold is effectively the number of points supporting the line
	// we start at 25% of diagonals
	cvCreateTrackbar("H Threshold", WINDOW_NAME, &param_h_threshold, 100, on_param_change);
	cvCreateTrackbar("V Threshold", WINDOW_NAME, &param_v_threshold, 100, on_param_change);

	img_edges 		= cvCreateImage(img_attrib_dim, 8, 1);
	img_h_edge_src 	= cvCreateImage(img_attrib_dim, 8, 1);
	img_v_edge_src 	= cvCreateImage(img_attrib_dim, 8, 1);
	img_color_dst 	= cvCreateImage(img_attrib_dim, 8, 3);

	mem_store = cvCreateMemStorage(0);

	h_smudge_kernel = cvCreateMat(5, 5, CV_32F);
	cvSet(h_smudge_kernel, cvRealScalar(0));
	for(int col_idx=0; col_idx < 5; col_idx++) {
		cvSet2D(h_smudge_kernel, 1, col_idx, cvRealScalar(1.0/4));
	}
	cvSet2D(h_smudge_kernel, 2, 2, cvRealScalar(0));

	v_smudge_kernel = cvCreateMat(5, 5, CV_32F);
	cvSet(v_smudge_kernel, cvRealScalar(0));
	for(int row_idx=0; row_idx < 5; row_idx++) {
		cvSet2D(v_smudge_kernel, row_idx, 1, cvRealScalar(1.0/4));
	}
	cvSet2D(v_smudge_kernel, 2, 2, cvRealScalar(0));
}

void destroy_gui() {
	cvDestroyWindow(WINDOW_NAME);
	cvReleaseImage(&img_read);
	cvReleaseImage(&img_src);
	cvReleaseImage(&img_smooth);
	cvReleaseImage(&img_edges);
	cvReleaseImage(&img_h_edge_src);
	cvReleaseImage(&img_v_edge_src);
	cvReleaseImage(&img_color_dst);

	cvReleaseMemStorage(&mem_store);

	cvReleaseMat(&h_smudge_kernel);
	cvReleaseMat(&v_smudge_kernel);
}


CvSeq* trim_spurious_v_lines(CvSeq *ret, int max_x, int max_shelves=MAX_ITEMS_PER_ROW) {
	if(ret->total <= 2) return ret;						// if we have just 2 lines or less, we can't process anything
	cvSeqSort(ret, compare_line_x_then_y, NULL);		// sort the lines so that we can calculate distance between them

	// remove close by lines as they are most probably spurious
	// find average y-distance
	// if y-dist is less than half of avg y-dist, put a line at the center of the two, remove one
	int dist_arr_cnt = 0;
	int *dist_arr = (int *)alloca((ret->total+1) * sizeof(int));

	// compare first line against edge of image
	CvPoint* first_line = (CvPoint*) cvGetSeqElem(ret, 0);
	if(first_line[0].x > max_x/max_shelves) dist_arr[dist_arr_cnt++] = first_line[0].x;

	for(int line_idx = 1; line_idx < ret->total; line_idx++) {
		CvPoint* line1 = (CvPoint*)cvGetSeqElem(ret, line_idx-1);
		CvPoint* line2 = (CvPoint*)cvGetSeqElem(ret, line_idx);
		int diff_dist = abs(line1[0].x - line2[0].x);

		if(diff_dist < max_x/max_shelves) {
			//printf("Ignoring line with distance %d\n", diff_dist);
			continue;
		}
		dist_arr[dist_arr_cnt++] = diff_dist;
		/*
		printf("diff %d. line1 %d,%d->%d,%d, line2 %d,%d->%d,%d\n", diff_dist,
				line1[0].x, line1[0].y, line1[1].x, line1[1].y,
				line2[0].x, line2[0].y, line2[1].x, line2[1].y);
		*/
	}

	// compare last line against edge of image
	CvPoint* last_line = (CvPoint*)cvGetSeqElem(ret, ret->total-1);
	if((max_x - last_line[0].x) > max_x/max_shelves) dist_arr[dist_arr_cnt++] = max_x - last_line[0].x;

	int x_median_dist = get_median_dist(dist_arr, dist_arr_cnt);
	//printf("Median vert shelve distance before trimming: %d. Considered: %d of %d lines\n", x_median_dist, dist_arr_cnt, ret->total);

	select_lines_by_separation(ret, x_median_dist, 0.75, false);
	//printf("Average vert shelve distance after trimming: %d. Num lines: %d\n", (int)avg_line_dist(ret, false), ret->total);

	return ret;
}


CvSeq* trim_spurious_h_lines(CvSeq *ret, int max_y, int max_shelves=MAX_SHELVES) {
	if(ret->total <= 2) return ret;

	cvSeqSort(ret, compare_line_y_then_x, NULL);

	// remove close by lines as they are most probably spurious
	// find average y-distance
	// if y-dist is less than half of avg y-dist, put a line at the center of the two, remove one
	int dist_arr_cnt = 0;
	int *dist_arr = (int *)alloca((ret->total+1) * sizeof(int));

	CvPoint* first_line = (CvPoint*)cvGetSeqElem(ret, 0);
	if(first_line[0].y > max_y/max_shelves) {
		dist_arr[dist_arr_cnt++] = first_line[0].y;
	}

	for(int line_idx = 1; line_idx < ret->total; line_idx++) {
		CvPoint* line1 = (CvPoint*)cvGetSeqElem(ret, line_idx-1);
		CvPoint* line2 = (CvPoint*)cvGetSeqElem(ret, line_idx);
		int diff_dist = abs(line1[0].y - line2[0].y);

		if(diff_dist < max_y/max_shelves) {
			//printf("Ignoring line with distance %d\n", diff_dist);
			continue;
		}
		dist_arr[dist_arr_cnt++] = diff_dist;
		/*
		printf("diff %d. line1 %d,%d->%d,%d, line2 %d,%d->%d,%d\n", diff_dist,
				line1[0].x, line1[0].y, line1[1].x, line1[1].y,
				line2[0].x, line2[0].y, line2[1].x, line2[1].y);
		*/
	}

	CvPoint* last_line = (CvPoint*)cvGetSeqElem(ret, ret->total-1);
	if((max_y - last_line[0].y) > max_y/max_shelves) {
		dist_arr[dist_arr_cnt++] = max_y - last_line[0].y;
	}

	int y_median_dist = get_median_dist(dist_arr, dist_arr_cnt);
	//printf("Median horz shelve distance before trimming: %d. Considered: %d of %d lines\n", y_median_dist, dist_arr_cnt, ret->total);

	select_lines_by_separation(ret, y_median_dist, 0.75, true);
	//printf("Average horz shelve distance after trimming: %d. Num lines: %d\n", avg_line_dist(ret, true), ret->total);

	return ret;
}

void process_h_v_edges() {
	// horizontal edge detect
	cvCvtColor(img_smooth, img_edges, CV_BGR2GRAY);
	cvLaplace(img_edges, img_edges, 3);

	cvFilter2D(img_edges, img_h_edge_src, h_smudge_kernel);
	cvThreshold(img_h_edge_src, img_h_edge_src, 200, 255, CV_THRESH_BINARY);

	// vertical edge detect
	cvFilter2D(img_edges, img_v_edge_src, v_smudge_kernel);
	cvThreshold(img_v_edge_src, img_v_edge_src, 200, 255, CV_THRESH_BINARY);
}

CvSeq* find_shelve_horz_lines(int width, int height) {
	CvSeq *ret = find_hough_lines(img_h_edge_src, mem_store, HOUGH_METHODS[param_hough_method],
			param_h_rho+1,
			2*CV_PI/(param_h_theta+1),
			width * (param_h_threshold+1) / 100,
			param_p1, param_p2, width);

	//printf("num horz lines before angle trim %d\n", ret->total);
	ret = select_lines_with_angle(ret, 0, CV_PI/18);
	//printf("num horz lines after angle trim %d\n", ret->total);

	// add a line for top and one for bottom
	CvPoint line_ends[2];
	line_ends[0].x = -width;
	line_ends[1].x = width;
	line_ends[0].y = line_ends[1].y = 0;
	cvSeqPush(ret, line_ends);
	line_ends[0].y = line_ends[1].y = height;
	cvSeqPush(ret, line_ends);

	int tot_before, tot_after;
	do {
		tot_before = ret->total;
		ret = trim_spurious_h_lines(ret, height);
		tot_after = ret->total;
	} while(tot_before != tot_after);

	printf("num horz lines: %d\n", ret->total);
	return ret;
}

CvSeq* find_shelve_vert_lines(int width, int height) {
	CvSeq *ret = find_hough_lines(img_v_edge_src, mem_store, HOUGH_METHODS[param_hough_method],
			param_v_rho+1,
			2*CV_PI/(param_v_theta+1),
			height * (param_v_threshold+1) / 100,
			param_p1, param_p2, height);

	//printf("num vert lines before angle trim %d\n", ret->total);
	ret = select_lines_with_angle(ret, CV_PI/2, CV_PI/72);
	//printf("num vert lines after angle trim %d\n", ret->total);

	// add a line for left and one for right
	CvPoint line_ends[2];
	line_ends[0].y = -height;
	line_ends[1].y = height;
	line_ends[0].x = line_ends[1].x = 0;
	cvSeqPush(ret, line_ends);
	line_ends[0].x = line_ends[1].x = width;
	cvSeqPush(ret, line_ends);


	int tot_before, tot_after;
	do {
		tot_before = ret->total;
		ret = trim_spurious_v_lines(ret, width);
		tot_after = ret->total;
	} while(tot_before != tot_after);

	printf("num vert lines: %d\n", ret->total);
	//print_lines(ret);

	return ret;
}

/*****************************************************
 * limit max size of image to be processed to 800x600
 *****************************************************/
void get_approp_size(IplImage *input, IplImage **output, CvSize &img_size, int &img_depth, int &img_channels) {
	CvSize ori_size = cvGetSize(input);
	img_depth = input->depth;
	img_channels = input->nChannels;

	printf("Image size: %d x %d\nDepth: %d\nChannels: %d\n", ori_size.width, ori_size.height, img_depth, img_channels);

	float div_frac_h = 600.0/((float)ori_size.height);
	float div_frac_w = 800.0/((float)ori_size.width);
	float div_frac = div_frac_w < div_frac_h ? div_frac_w : div_frac_h;
	if(div_frac > 1) div_frac = 1;

	img_size.height = ori_size.height * div_frac;
	img_size.width = ori_size.width * div_frac;

	*output = cvCreateImage(img_size, img_depth, img_channels);
	cvResize(input, *output);
}

int main(int argc, char** argv) {
	if(argc < 2) {
		printf("Usage: %s <image>\n", argv[0]);
		return 1;
	}

	img_read = cvLoadImage(argv[1]);
	if(NULL == img_read) {
		printf("Error loading image %s\n", argv[1]);
		return -1;
	}

	get_approp_size(img_read, &img_src, img_attrib_dim, img_attrib_depth, img_attrib_channels);
	create_gui();

	int col_idx = 0;

	img_smooth = cvCreateImage(img_attrib_dim, img_attrib_depth, img_attrib_channels);

	while(true) {
		if(param_changed) {
			param_changed = false;
			smooth_brightness_contrast(img_src, img_smooth, param_brightness_factor-50, param_contrast_factor);

			cvClearMemStorage(mem_store);

			process_h_v_edges();
			cvCopy(img_src, img_color_dst);

			CvSeq* lines = find_shelve_horz_lines(img_attrib_dim.width, img_attrib_dim.height);

			// rotate color for the lines
			if(LINE_DRAW_NUM_COLORS == (++col_idx)) col_idx = 0;

			// for each image segment, find vertical lines
			CvRect roi_rect;
			roi_rect.x = 0;
			roi_rect.y = 0;
			roi_rect.width = img_attrib_dim.width;
			int max_y = img_attrib_dim.height;

			for(int line_idx = 0; line_idx < lines->total; line_idx++) {
				CvPoint* line = (CvPoint*)cvGetSeqElem(lines, line_idx);

				// skip if the line seems to be the top or bottom border
				if(line[0].y < max_y/MAX_SHELVES) {
					//printf("skipping beginning line with y[%d] max_y[%d]\n", line[0].y, max_y);
					continue;
				}

				roi_rect.height = line[0].y - roi_rect.y;
				cvSetImageROI(img_v_edge_src, roi_rect);

				// process
				CvSeq* v_lines = find_shelve_vert_lines(roi_rect.width, roi_rect.height);
				for(int line_idx = 0; line_idx < v_lines->total; line_idx++) {
					CvPoint* one_v_line = (CvPoint*)cvGetSeqElem(v_lines, line_idx);
					one_v_line[0].y = roi_rect.y;
					one_v_line[1].y = roi_rect.y + roi_rect.height;
				}
				draw_lines(v_lines, img_color_dst, col_idx);

				// prepare rect for the next slot
				roi_rect.y += roi_rect.height;
				cvResetImageROI(img_v_edge_src);
			}

			draw_lines(lines, img_color_dst, col_idx);

			if(0 == param_disp_image) 		cvShowImage(WINDOW_NAME, img_src);
			else if(1 == param_disp_image) 	cvShowImage(WINDOW_NAME, img_smooth);
			else if(2 == param_disp_image) 	cvShowImage(WINDOW_NAME, img_edges);
			else if(3 == param_disp_image) 	cvShowImage(WINDOW_NAME, img_h_edge_src);
			else if(4 == param_disp_image) 	cvShowImage(WINDOW_NAME, img_v_edge_src);
			else if(5 == param_disp_image) 	cvShowImage(WINDOW_NAME, img_color_dst);
		}

		char c = cvWaitKey(500);
		if(27 == c) break;
	}

	destroy_gui();
}

