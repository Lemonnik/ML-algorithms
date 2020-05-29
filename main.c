#include <gtk/gtk.h>
#include "handlers.h"

int main (int argc, char **argv) {
	GtkBuilder *builder;
	GObject *window, *drawing_area,
			*next, *entry, *empty, *completed, *points;
	Widgets *w = g_slice_new0(Widgets);
	GError *error = NULL;

	gtk_init(&argc, &argv);
	builder = gtk_builder_new();
	if (gtk_builder_add_from_file (builder, INTERFACE, &error) == 0) {
		g_printerr ("Error loading file: %s\n", error->message);
		g_clear_error (&error);
		return 1;
	}
	window = gtk_builder_get_object (builder, "window");
	drawing_area = gtk_builder_get_object (builder, "drawing_area");
	next = gtk_builder_get_object(builder, "next");
	entry = gtk_builder_get_object (builder, "input");
	empty = gtk_builder_get_object(builder, "empty");
	completed = gtk_builder_get_object(builder, "completed");
	points = gtk_builder_get_object(builder, "points");

	w->entry = GTK_WIDGET(entry);
	w->empty = GTK_WIDGET(empty);
	w->completed = GTK_WIDGET (completed);
	w->points = GTK_WIDGET (points);
	w->drawing_area = GTK_WIDGET(drawing_area);

	gtk_builder_connect_signals(builder, NULL);
	g_signal_connect(next, "clicked", G_CALLBACK(on_next_clicked), w);

	gtk_widget_set_events (GTK_WIDGET(drawing_area),
			gtk_widget_get_events (GTK_WIDGET(drawing_area)) | GDK_BUTTON_PRESS_MASK);


	g_object_unref(G_OBJECT(builder));
	gtk_widget_show_all(GTK_WIDGET(window));
	gtk_main();
	g_slice_free (Widgets, w);
  return 0;
}
