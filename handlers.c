#include <gtk/gtk.h>
#include <stdlib.h>
#include <string.h>
#include "handlers.h"

cairo_surface_t *surface = NULL;
extern Points *points;

void clear_surface () {
  cairo_t *cr;

  cr = cairo_create (surface);
  cairo_set_source_rgb (cr, 1, 1, 1);
  cairo_paint (cr);

  cairo_destroy (cr);
}

gboolean on_drawing_area_configure_event (GtkWidget *widget,
		GdkEventConfigure *event, gpointer data) {
	if (surface) {
		cairo_surface_destroy (surface);
	}
	if (points) {
		Points_clear();
	}
	surface = gdk_window_create_similar_surface (gtk_widget_get_window (widget),
											   CAIRO_CONTENT_COLOR,
											   gtk_widget_get_allocated_width (widget),
											   gtk_widget_get_allocated_height (widget));
	clear_surface ();
	return TRUE;
}

gboolean on_drawing_area_draw (GtkWidget *widget, cairo_t *cr, gpointer data) {
	cairo_set_source_surface (cr, surface, 0, 0);
	cairo_paint (cr);

	return FALSE;
}

gboolean on_drawing_area_button_press_event (GtkWidget *widget,
		GdkEventButton *event, gpointer data) {
  if (surface == NULL)
    return FALSE;

  if (event->button == 1) {
      push_back (event->x, event->y);
      draw_point (widget, &(points->p[points->n - 1]), NULL);
  }
  else if (event->button == 3){
      clear_surface ();
      gtk_widget_queue_draw (widget);
      Points_clear ();
  }

  return TRUE;
}


void draw_point (GtkWidget *widget, Point *p, Color *c) {
	cairo_t *cr;
	cr = cairo_create (surface);

	cairo_set_line_width (cr, 2);
	cairo_arc (cr, p->x, p->y, S, 0, 2 * G_PI);
	cairo_stroke_preserve (cr);

	if (points->eps < E) {
		find_closest (p);
	}
	cairo_set_source_rgb (cr, p->clr.r, p->clr.g, p->clr.b);
	cairo_fill (cr);

	cairo_destroy (cr);
	gtk_widget_queue_draw_area (widget, p->x - (S + 1), p->y - (S + 1),
						 2 * (S + 1), 2 * (S + 1));
}

void draw_centroid (GtkWidget *widget, Point *p) {
	cairo_t *cr;
	cr = cairo_create (surface);

	cairo_set_line_width (cr, 3);
	cairo_set_source_rgb (cr, p->clr.r, p->clr.g, p->clr.g);
	cairo_move_to (cr, p->x - S, p->y - S);
	cairo_line_to (cr, p->x + S, p->y + S);
	cairo_move_to (cr, p->x + S, p->y - S);
	cairo_line_to (cr, p->x - S, p->y + S);
	cairo_stroke (cr);
	cairo_destroy (cr);
	gtk_widget_queue_draw_area (widget, p->x - 2 * S, p->y - 2 * S,
								p->x + 2 * S, p->y + 2 * S);
}

void draw_points (GtkWidget *widget) {
	int i;
	for (i = 0; i < points->n; i++) {
		draw_point (widget, &(points->p[i]), &(points->p[i].clr));
	}
}

void draw_centroids (GtkWidget *widget) {
	int i;
	for (i = 0; i < points->k; i++) {
		draw_centroid (widget, &(points->C[i]));
	}
}

gboolean on_clear_clicked (GtkWidget *widget, gpointer data) {
	if (surface == NULL) {
		return FALSE;
	}
	clear_surface ();
	gtk_widget_queue_draw (widget);
	Points_clear ();
	return TRUE;
}


void redraw_clusters (GtkWidget *widget) {
	clear_surface ();
	gtk_widget_queue_draw (widget);
	draw_centroids (widget);
	draw_points (widget);
}


void erase_color () {
	int i;
	Color w = {1, 1, 1};
	for (i = 0; i < points->n; i++) {
		points->p[i].clr = w;
	}
}

void init_centroids (GtkWidget *widget, int k) {
	if (k != points->k) {
		clear_surface ();
		gtk_widget_queue_draw (widget);
		erase_color ();
		draw_points (widget);
	}
	set_centroids (k);
	draw_centroids (widget);
}

bool check_input (Widgets *w, int k) {
	if (k <= 0) {
		gtk_dialog_run (GTK_DIALOG (w->empty));
		return FALSE;
	}
	if (points == NULL || points->n < k) {
		gtk_dialog_run (GTK_DIALOG (w->points));
		return FALSE;
	}
	return TRUE;
}


gboolean on_next_clicked (GtkWidget *next, Widgets *w) {
	int k;
	const gchar *input = gtk_entry_get_text (GTK_ENTRY (w->entry));

	sscanf (input, "%d", &k);
	if (!check_input (w, k)) {
		return FALSE;
	}
	/* one iteration of k-means */
	if (points->eps > E) {
		/* centroids selection as separate step */
		if (points->C == NULL || k != points->k) {
			init_centroids (w->drawing_area, k);
		}
		else {
			assign_clusters (); /* assign points to the closest centroids */
			update_centroids (); /* update centroids coordinates */
			redraw_clusters (w->drawing_area); /* reassign points to the new closest centroids */
		}
	}
	else {
		gtk_dialog_run (GTK_DIALOG (w->completed));
	}
	return TRUE;
}
