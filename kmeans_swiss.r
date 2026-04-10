# ============================================================
# K-Means Clustering Analysis — swiss Dataset
#
# Research Question:
#   "How do Swiss provinces naturally group together, and how
#    do those groupings relate to socioeconomic, educational,
#    and demographic characteristics?"
#
# Dataset: swiss (built-in R)
#   47 French-speaking Swiss provinces, circa 1888
#   6 numeric variables:
#     Fertility        – standardised fertility measure
#     Agriculture      – % males in agriculture
#     Examination      – % army recruits with highest exam score
#     Education        – % beyond primary school
#     Catholic         – % Catholic (vs Protestant)
#     Infant.Mortality – infant deaths per 1000 births
#
# Packages: tidyverse, broom, patchwork
# ============================================================

library(tidyverse)
library(broom)
library(patchwork)


# ── 1. Prepare the data ──────────────────────────────────────

# All 6 variables are numeric — scale all of them
swiss_scaled <- swiss %>%
  rownames_to_column("province") %>%
  mutate(across(Fertility:Infant.Mortality, scale))

# The 6 clustering variables (as a plain matrix for kmeans)
clust_vars <- swiss_scaled %>%
  select(Fertility, Agriculture, Examination,
         Education, Catholic, Infant.Mortality)


# ── 2. Elbow Method — choose k ───────────────────────────────

set.seed(42)

elbow_data <- tibble(k = 1:9) %>%
  mutate(
    km     = map(k, ~ kmeans(clust_vars, centers = .x, nstart = 25)),
    glance = map(km, glance)
  ) %>%
  unnest(glance)

# Inspect elbow_data$tot.withinss — we'll choose k = 3


# ── 3. Fit final model (k = 3) ───────────────────────────────

km_fit <- kmeans(clust_vars, centers = 3, nstart = 25)

# Centroids (one row per cluster, one column per variable)
centroids <- tidy(km_fit)
print(centroids)

# Attach cluster to original unscaled data
swiss_clustered <- swiss %>%
  rownames_to_column("province") %>%
  mutate(cluster = factor(km_fit$cluster,
                          labels = paste("Cluster", 1:3)))


# ── 4. PCA for the 2-D scatter (Panel B) ─────────────────────
# With 6 variables we can't pick a single pair of axes and trust it
# to represent full separation. PCA gives the best 2-D summary.

pca_result <- prcomp(clust_vars)          # already scaled, so no re-scaling

# % variance explained by PC1 and PC2
pct_var <- round(100 * pca_result$sdev^2 /
                   sum(pca_result$sdev^2), 1)

# PC scores per province
pca_scores <- as_tibble(pca_result$x[ , 1:2]) %>%
  bind_cols(swiss_clustered %>% select(province, cluster))

# PC loadings (arrows for biplot overlay)
loadings <- as_tibble(pca_result$rotation[ , 1:2],
                      rownames = "variable") %>%
  mutate(across(c(PC1, PC2), ~ .x * 3))   # scale arrows for visibility


# ── 5. Long-format for parallel coordinates (Viz 2) ──────────

swiss_long <- swiss_scaled %>%
  mutate(cluster = factor(km_fit$cluster,
                          labels = c("Cluster 1",
                                     "Cluster 2",
                                     "Cluster 3"))) %>%
  pivot_longer(
    cols      = Fertility:Infant.Mortality,
    names_to  = "variable",
    values_to = "z_score"
  ) %>%
  mutate(variable = factor(variable,
                           levels = c("Fertility", "Agriculture",
                                      "Examination", "Education",
                                      "Catholic", "Infant.Mortality"),
                           labels = c("Fertility", "Agri.", "Exam",
                                      "Educ.", "Catholic", "Infant\nMort.")))

cluster_means <- swiss_long %>%
  group_by(cluster, variable) %>%
  summarise(z_score = mean(z_score), .groups = "drop")


# ── 6. Colour palette ────────────────────────────────────────

pal <- c("Cluster 1" = "#2196F3",
         "Cluster 2" = "#F44336",
         "Cluster 3" = "#4CAF50")


# ── 7. VISUALIZATION 1 — Compound multi-panel (patchwork) ────
#
#   Layout:
#     [ Panel A: Elbow  |  Panel B: PCA scatter + biplot arrows ]
#     [       Panel C: Box plots — all 6 variables by cluster   ]

# ── Panel A: Elbow plot ──
p_elbow <- ggplot(elbow_data, aes(x = k, y = tot.withinss)) +
  geom_line(colour = "grey55", linewidth = 0.8) +
  geom_point(size = 3.5, colour = "#3F51B5") +
  geom_vline(xintercept = 3, linetype = "dashed",
             colour = "#F44336", linewidth = 0.7) +
  annotate("text", x = 3.2,
           y = max(elbow_data$tot.withinss) * 0.88,
           label = "k = 3\nchosen", hjust = 0,
           size = 3.2, colour = "#F44336") +
  scale_x_continuous(breaks = 1:9) +
  labs(
    title    = "A  Elbow Method",
    subtitle = "Total within-cluster SS across k = 1–9",
    x = "Number of clusters (k)",
    y = "Total Within-Cluster SS"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey40"))

# ── Panel B: PCA scatter with biplot arrows ──
p_pca <- ggplot(pca_scores,
                aes(x = PC1, y = PC2, colour = cluster)) +
  # Province points
  geom_point(size = 2.8, alpha = 0.85) +
  geom_text(aes(label = province), size = 2.0, vjust = -0.8,
            show.legend = FALSE, check_overlap = TRUE) +
  # Biplot arrows (loadings)
  geom_segment(data = loadings,
               aes(x = 0, y = 0, xend = PC1, yend = PC2),
               inherit.aes = FALSE,
               arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
               colour = "grey30", linewidth = 0.5) +
  geom_text(data = loadings,
            aes(x = PC1 * 1.12, y = PC2 * 1.12, label = variable),
            inherit.aes = FALSE,
            size = 2.8, colour = "grey20", fontface = "italic") +
  scale_colour_manual(values = pal) +
  labs(
    title    = "B  PCA Scatter — Cluster Separation",
    subtitle = glue::glue("PC1 explains {pct_var[1]}% of variance, ",
                          "PC2 explains {pct_var[2]}%"),
    x       = glue::glue("PC1 ({pct_var[1]}% variance)"),
    y       = glue::glue("PC2 ({pct_var[2]}% variance)"),
    colour  = "Cluster"
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title    = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 9, colour = "grey40"),
        legend.position = "bottom",
        legend.title  = element_text(size = 9),
        legend.text   = element_text(size = 9))

# ── Panel C: Box plots — all 6 variables, faceted by variable ──
# Reshape original unscaled data to long format for boxplots
swiss_box_long <- swiss_clustered %>%
  pivot_longer(
    cols      = Fertility:Infant.Mortality,
    names_to  = "variable",
    values_to = "value"
  ) %>%
  mutate(variable = factor(variable,
                           levels = c("Fertility", "Agriculture",
                                      "Examination", "Education",
                                      "Catholic", "Infant.Mortality"),
                           labels = c("Fertility", "Agriculture",
                                      "Examination", "Education",
                                      "Catholic", "Infant Mortality")))

p_box <- ggplot(swiss_box_long,
                aes(x = cluster, y = value,
                    fill = cluster, colour = cluster)) +
  geom_boxplot(alpha = 0.35, outlier.shape = NA, linewidth = 0.5) +
  geom_jitter(width = 0.18, size = 1.5, alpha = 0.7) +
  facet_wrap(~ variable, scales = "free_y", nrow = 1) +
  scale_fill_manual(values   = pal) +
  scale_colour_manual(values = pal) +
  labs(
    title    = "C  Variable Distributions by Cluster",
    subtitle = "Raw (unscaled) values — each panel = one of the 6 variables",
    x = NULL, y = "Value"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    plot.title       = element_text(face = "bold", size = 12),
    plot.subtitle    = element_text(size = 9, colour = "grey40"),
    strip.text       = element_text(face = "bold", size = 9),
    strip.background = element_rect(fill = "grey95", colour = NA),
    axis.text.x      = element_text(size = 7.5, angle = 15, hjust = 1),
    legend.position  = "none",
    panel.grid.minor = element_blank()
  )

# ── Compose with patchwork ──
viz1 <- (p_elbow | p_pca) / p_box +
  plot_layout(heights = c(1.1, 1)) +
  plot_annotation(
    title    = "K-Means Clustering of Swiss Provinces (k = 3)",
    subtitle = paste(
      "Research Question: How do Swiss provinces naturally group together,",
      "and how do those groupings relate to socioeconomic,",
      "educational, and demographic characteristics?"
    ),
    caption  = "Data: swiss (R built-in, 1888) | Packages: tidyverse, broom, patchwork",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle = element_text(size = 10, colour = "grey30", hjust = 0),
      plot.caption  = element_text(size = 8,  colour = "grey50", hjust = 1)
    )
  )

#ggsave("viz1_multipanel_swiss.png", viz1,
       #width = 14, height = 10, dpi = 180, bg = "white")
#message("Saved → viz1_multipanel_swiss.png")
viz1

# ── 8. VISUALIZATION 2 — Faceted parallel coordinates ────────
#   facet_wrap(~ cluster): one panel per cluster
#   Each thin line = one province across all 6 z-scored variables
#   Bold line = cluster mean profile

violin_means <- swiss_box_long %>%
  group_by(cluster, variable) %>%
  summarise(mean_val = mean(value), .groups = "drop")

viz2 <- ggplot(swiss_box_long,
               aes(x = cluster, y = value,
                   fill = cluster, colour = cluster)) +
  
  # Violin — shows the full distribution shape per cluster
  geom_violin(alpha = 0.35, linewidth = 0.5, trim = FALSE) +
  
  # Jittered raw points — every individual province visible
  geom_jitter(width = 0.10, size = 1.8, alpha = 0.65,
              show.legend = FALSE) +
  
  # White dot = cluster mean, sits on top of violin
  geom_point(data = violin_means,
             aes(x = cluster, y = mean_val, colour = cluster),
             inherit.aes = FALSE,
             shape = 21, size = 4,
             fill = "white", stroke = 1.8,
             show.legend = FALSE) +
  
  # One facet per variable; free_y so each panel uses its own scale
  facet_wrap(~ variable, scales = "free_y", nrow = 2, ncol = 3) +
  
  scale_fill_manual(values   = pal) +
  scale_colour_manual(values = pal) +
  
  labs(
    title    = "Distribution of Raw Variable Values by Cluster",
    subtitle = paste(
      "Each facet = one variable on its own scale (no standardisation).",
      "Violin = distribution shape · points = individual provinces · white dot = cluster mean."
    ),
    x       = "Cluster",
    y       = "Raw Value",
    fill    = "Cluster",
    caption = "Data: swiss (R built-in, 1888) | Packages: tidyverse, broom, patchwork"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0),
    plot.subtitle    = element_text(size = 10, colour = "grey30", hjust = 0),
    plot.caption     = element_text(size = 8,  colour = "grey50", hjust = 1),
    strip.text       = element_text(face = "bold", size = 11),
    strip.background = element_rect(fill = "grey95", colour = NA),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom",
    legend.title     = element_text(size = 10),
    axis.text.x      = element_text(size = 9)
  )

#ggsave("viz2_faceted_swiss.png", viz2,
       #width = 13, height = 5.5, dpi = 180, bg = "white")
#message("Saved → viz2_faceted_swiss.png")
viz2

# ── 9. Cluster summary table ─────────────────────────────────

summary_tbl <- swiss_clustered %>%
  group_by(cluster) %>%
  summarise(
    n                = n(),
    avg_fertility    = round(mean(Fertility),        1),
    avg_agriculture  = round(mean(Agriculture),      1),
    avg_examination  = round(mean(Examination),      1),
    avg_education    = round(mean(Education),        1),
    avg_catholic     = round(mean(Catholic),         1),
    avg_infant_mort  = round(mean(Infant.Mortality), 1),
    .groups = "drop"
  )

cat("\n── Cluster Summary (unscaled means) ─────────────────\n")
print(summary_tbl)
cat("\nNote: cluster labels (1/2/3) are arbitrary numbers from kmeans().\n")
cat("Interpret them by inspecting the centroid values above.\n\n")
