#ifndef K_MEANS_H_
#define K_MEANS_H_
/* sum of absolute differences between old and new centroids positions */
#define E 1

typedef struct Color {
	double r, g, b;
} Color;

typedef struct {
	double x, y;
	/* number of cluster that point belongs to */
	int cluster;
	Color clr;
} Point;

typedef struct {
	Point *p; /* points */
	Point *C; /* centroids */
	int n; /* number of points */
	int k; /* number of clusters */
	/* sum of absolute differences between previous and current centroids positions,
	 * (as criterion of termination) */
	double eps;
} Points;

typedef enum {false, true} bool;

/* 'constructor' for Points */
Points* Points_init ();
/* 'destructor' for Points */
void Points_clear();

/* append new Point to Points */
void push_back (const int, const int);
/* randomly select (without replacement) k centroid indices from 0 to n-1 */
void random_sampling (int*, const int k, const int n);
/* set random color for cluster */
Color random_color ();
/* select and save k centroids into Points->C */
void set_centroids (const int k);
/* euclidean distance */
double dist (const Point*, const Point*);
/* find the closest (by euclidean distance) centroid for point */
void find_closest (Point*);
/* find and assign the index of closest centroid to Point->cluster */
void assign_clusters ();
/* find mean of all cluster related points */
Point find_mean (const int);
/* recompute centroids coordinates according to cluster related points */
void update_centroids ();

#endif /* K_MEANS_H_ */
