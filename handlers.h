#ifndef HANDLERS_H_
#define HANDLERS_H_

#include <gtk/gtk.h>
#include "k_means.h"

/* path to xml */
#define INTERFACE "interface.ui"
/* point size */
#define S 3

typedef struct {
	GtkWidget *entry, *empty, *completed,
				*points, *drawing_area;
} Widgets;

void clear_surface ();
gboolean on_drawing_area_configure_event (GtkWidget*,
		GdkEventConfigure*, gpointer);
gboolean on_drawing_area_draw (GtkWidget*, cairo_t*, gpointer);
gboolean on_drawing_area_button_press_event (GtkWidget*, GdkEventButton*,
        gpointer);
void draw_point (GtkWidget*, Point*, Color *c);
void draw_centroid (GtkWidget*, Point*);
void draw_points (GtkWidget*);
void draw_centroids (GtkWidget*);
void redraw_clusters (GtkWidget*);
void init_centroids (GtkWidget *, int);
void erase_color ();
gboolean on_clear_clicked (GtkWidget*, gpointer);
gboolean on_next_clicked (GtkWidget*, Widgets*);

#endif /* HANDLERS_H_ */
