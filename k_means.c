#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include "k_means.h"

Points *points = NULL;


Points* Points_init () {
	Points *new = malloc(sizeof(Points));

	new->n = 0;
	new->p = NULL;

	new->k = 0;
	new->C = NULL;

	new->eps = DBL_MAX;
	return new;
}

void push_back (const int x, const int y) {
	if (!points) {
		points = Points_init ();
	}
	Color clr = {1., 1., 1.};
	Point last = {.x = (double)x, .y = (double)y,
									.cluster = 0, .clr = clr};
	int size = points->n + 1;

	points->p = realloc (points->p, size * sizeof(Point));
	points->p[points->n++] = last;
}


void random_sampling (int *idx, const int k, const int n) {
	int i;
	bool *used = malloc(n * sizeof(bool));
	memset(used, false, n * sizeof(bool));
	for (i = 0; i < k; i++) {
		do {
			idx[i] = rand() % n;
		} while (used[idx[i]]);
		used[idx[i]] = true;
	}
	free (used);
}

Color random_color () {
	double r, g, b;
	r = rand() * 1.0 / RAND_MAX;
	g = rand() * 1.0 / RAND_MAX;
	b = rand() * 1.0 / RAND_MAX;

	Color c = {.r = r, .g = g, .b = b};
	return c;
}

void set_centroids (const int k) {
	int i, idx[k];
	points->k = k;

	if (points->C != NULL) {
		free (points->C);
		points->C = NULL;
		points->eps = DBL_MAX;
	}
	random_sampling (idx, k, points->n);
	points->C = malloc (k * sizeof(Point));
	srand (time (0));
	for (i = 0; i < k; i++) {
		points->C[i] = points->p[idx[i]];
		points->C[i].clr = random_color ();
	}
}


double dist (const Point *a, const Point *b) {
	return sqrt(pow(a->x - b->x, 2) + pow(a->y - b->y, 2));
}

void find_closest (Point *p) {
	int i;
	double r, min_r = DBL_MAX;
	for (i = 0; i < points->k; i++) {
		r = dist(p, &(points->C[i]));
		if (r < min_r) {
			min_r = r;
			p->cluster = i;
			p->clr = points->C[i].clr;
		}
	}
}

void assign_clusters () {
	int i;
	for (i = 0; i < points->n; i++) {
		find_closest (&(points->p[i]));
	}
}

Point find_mean (const int c) {
	int i, k;
	Point p = {.x = 0., .y = 0., .cluster = c, .clr = points->C[c].clr};
	k = 0;
	for (i = 0; i < points->n; i++) {
		if (points->p[i].cluster == c) {
			p.x += points->p[i].x;
			p.y += points->p[i].y;
			k++;
		}
	}
	p.x /= k, p.y /= k;

	return p;
}

void update_centroids () {
	int i;
	double eps = 0.;
	Point p;
	for (i = 0; i < points->k; i++) {
		p = find_mean (i);
		eps += abs(p.x - points->C[i].x) + abs(p.y - points->C[i].y);
		points->C[i] = p;
	}
	points->eps = eps;
}


void Points_clear () {
	if (points) {
		free (points->p);
		points->p = NULL;

		if (points->C) {
			free (points->C);
			points->C = NULL;
		}
		free (points);
		points = NULL;
	}
}
