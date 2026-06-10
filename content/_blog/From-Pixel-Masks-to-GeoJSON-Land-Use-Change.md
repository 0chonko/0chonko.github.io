---
title: "From Pixel Masks to GeoJSON: Building a Land Use Change Pipeline in Processing"
date: 2021-08-01
description: "How a bachelor thesis tool turned noisy satellite change-detection tiles into web-ready polygons, and what that taught me about data shape before visualization."
---

Most geospatial dashboards fail quietly. The map loads, the layers toggle, and nobody asks how many millions of points sat behind the first Kepler upload.

For my bachelor graduation project at the **University of Twente** (Creative Technology, 2021), I worked with **RISE** in Cyprus on a parallel line of research: weakly supervised **Siamese neural networks** that compare Planet satellite image pairs and emit **change detection (CD) maps** as 256×256 binary masks between **Nicosia** and **Larnaca**. My job was not the model itself. It was the uncomfortable middle layer: turn those masks into something policymakers could actually explore on the web, then relate change patterns to demographic data from Copernicus CORINE and Michael Bauer Research via **ArcGIS**.

The central question in the thesis was direct: **what correlations show up most often between land use change and demographic factors?**

---

## The Data Shape Problem

Each CD tile is a dense pixel decision: image A vs image B at 256×256 resolution, roughly **3 m** ground sampling. A two-week comparison over a city slice can mean **tens of millions** of classified pixels. Kepler.gl and ArcGIS Online are good tools, but they are not happy when you throw raw mask files at them and ask for time-series playback.

The thesis measured the gap concretely: after clustering and polygon simplification, the exported **GeoJSON** landed around **10 to 20× smaller** than the original CD text masks, while keeping the spatial story intact.

That size drop was not cosmetic. It was the difference between "upload times out" and "scrub a timeline."

---

## What the Tool Actually Does

The analysis tool is a mode-switching Processing app (keyboard shortcuts flip between `scatter`, `line`, `squarify`, `rawData`, `imgCombiner`, contour export, etc.). Early modes were exploratory: plot demographic CSV columns against coordinates, draw connection lines between settlements sharing attributes, grid-aggregate values with `squarefy`. Useful for ideation, not for shipping.

The production path that mattered for the thesis looked like this:

```text
CD mask (Output/) + lat/lon grid (Lat_mask_entire/)
        |
        v
OpenCV findContours  -->  pixel contours (DELIM-separated clusters)
        |
        v
index = y * res + x  -->  lookup geographic coordinate
        |
        v
sort vertices (centroid angles)  -->  concave hull (k-NN)
        |
        v
GeoJSON FeatureCollection  -->  Kepler.gl / ArcGIS / Tableau
```

### Pixel index to coordinate

Every mask file lines up with a companion `Lat_mask_entire` grid. The code maps contour pixels back to geographic positions with a flat index:

```text
currentIndex = floor(y) * res + floor(x)
```

That sounds trivial until you are juggling hundreds of patches named like `Merge_30032021_0_768.txt` across a 11776×13312 parent image. `fileNameGenerator()` exists because the naming scheme *is* the spatial index: step 256 on x until the row ends, then reset x and step y.

### Contours, ordering, hulls

**OpenCV** (`RETR_EXTERNAL`, `CHAIN_APPROX_NONE`) extracts outer boundaries from each mask tile. Contour points arrive in scan order, which is wrong for filled polygons in Kepler. The sketch re-sorts vertices by angle around a centroid (`geoJSONrearrangeVertexes`), then runs a **concave hull** (`testHull`, k-nearest) so clusters stay tight without the bloated geometry of a convex hull.

Each closed cluster becomes a GeoJSON `Feature` with properties for timestamp, cluster id, and a change class among:

* reforestation
* deforestation
* construction
* newInfrastructure
* cropChange
* mine

Some labels were simulated when the final classified dataset had not landed yet. That was a deadline trade-off, not a claim about ground truth.

### Side outputs

The same codebase also generated **CSV** exports for ArcGIS hotspot analysis, **side-by-side JPEG strips** for manual labeling (`printImagesForLabeling`), and early **mil.nga.sf.geojson** experiments before settling on the FeatureCollection writer in `multiPolyGeoJson()`.

---

## What We Saw Once It Rendered

With simplified polygons in Kepler, patterns showed up fast:

* **Rural vs urban:** agricultural belts around Nicosia and Larnaca produced larger contiguous change blobs; inside cities, detections were smaller but more frequent (construction sites, infill).
* **Time playback:** February to March 2021 comparisons showed change propagating east-to-west between the two cities, which was easier to read as a Kepler timeline than as static screenshots.
* **Hotspot checks in ArcGIS:** converting polygons to centroids unlocked **Getis-Ord Gi*** hotspot maps. February looked "colder" (rarer significant clusters); March shifted hotter across much of the study area. That matched the eyeball test from Kepler, which was reassuring because the CD maps still carried plenty of false positives.

Overlaying **CORINE 2019** land cover added context (color by cover class, elevation by cluster area in 3D Kepler views), with the obvious caveat: two years of lag means you cannot infer urban growth from CORINE alone.

Demographic correlation work (population, income, purchasing power via MBR enrichment) leaned on ArcGIS interpolation and filtering. Sparse demographic grids next to 3 m CD data gets misleading fast if you do not clip and filter interpolated surfaces. I dropped large low-confidence cells (the thesis example: ~69 km² estimated from fewer than 10 points).

---

## What This Connects To

Looking back from later work on **Snellius** and MPI/network profiling, the same instinct keeps showing up: the interesting failure is rarely the algorithm in isolation. It is the **interface** between a high-rate producer (DNN masks, RDMA counters, GPU telemetry) and a human-facing tool that expects a different grain of data.

I have not re-run this pipeline on modern vector tiles or COGs. If I did, I would probably push contour + hull steps into a geospatial notebook or GDAL workflow and keep Processing for exploration only. The geometry lessons (ordering, hull tightness, delimiter-separated clusters) still transfer.

---

## Practical Takeaways

1. **Simplify before the dashboard.** Web GL map layers want polygons or tiled aggregates, not per-pixel truth. Measure upload size and scrub latency, not just visual aesthetics.
2. **Treat filenames as spatial metadata** when grids are regular. A consistent `Merge_{date}_{y}_{x}` pattern saved hours of manual stitching.
3. **Validate geometry before properties.** One open polygon or scrambled vertex order breaks fill in every downstream tool the same way.
4. **Keep raw and derived paths separate.** Masks for science, GeoJSON for storytelling, CSV centroids for ArcGIS stats. Mixing them early creates format thrash.
5. **Name your uncertainty.** False positives in the CD maps and simulated class labels were real limitations. The hotspot trends were still directionally useful because rural/urban splits matched expectations.

---

### References

* G. Savchenko, *Visualizing Land Use Change* (B.Sc. thesis, University of Twente, July 2021). [PDF](/Land_Use_Change_analysis_tool/Bachelor-thesis.pdf)
* Kalita et al., *Land Use Change Detection Using Deep Siamese Neural Networks and Weakly Supervised Learning* (CAIP 2021). [PDF](https://superworld.cyens.org.cy/papers/land_change_CAIP21.pdf)
* Tool source: [`Land_Use_Change_analysis_tool`](https://github.com/0chonko/Land_Use_Change_analysis_tool)
* Full list: [All references](/readinglist/).

---

