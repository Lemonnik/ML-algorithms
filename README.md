GTK-based k-means algorithm implementation for 2d data. For compilation run

```
gcc `pkg-config --cflags gtk+-3.0` -o kmeans main.c handlers.c k_means.c -lm -rdynamic `pkg-config --libs gtk+-3.0`
```
