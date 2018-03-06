/*
 * cv_utils.h
 *
 *  Created on: Dec 7, 2012
 *      Author: tan
 *  Related blog post at:
 *  http://sidekick.windforwings.com/2012/12/opencv-separating-items-on-store-shelves.html
 */

#ifndef CV_UTILS_H_
#define CV_UTILS_H_

#define LINE_DRAW_NUM_COLORS 3

void print_lib_version();

int compare_line_y_then_x(const void *_l1, const void *_l2, void* userdata);
int compare_line_x_then_y(const void *_l1, const void *_l2, void* userdata);
int compare_int(const void *i1, const void *i2);

double avg_line_dist(CvSeq *lines, bool is_y);
int get_median_dist(int *dist_arr, int dist_arr_cnt);

CvPoint * rho_theta_to_line(float *rho_theta, CvPoint *line, int max_dim);
CvSeq* select_lines_with_angle(CvSeq *ret, double angle, double accuracy);
CvSeq* select_lines_by_separation(CvSeq *ret, int sep, float accuracy, bool is_y);

void print_lines(CvSeq *lines);
void smooth_brightness_contrast(IplImage *img, IplImage *img_result, int param_brightness_factor, int param_contrast_factor);

void draw_lines(CvSeq* lines, IplImage *img_color_dst, int col_idx);
CvSeq * find_hough_lines(IplImage *img, CvMemStorage *mem_store, int hough_method, double rho, double theta, int threshold, double p1, double p2, int max_dim);

#endif /* CV_UTILS_H_ */
