const qs = (selector, ctx = document) => ctx.querySelector(selector);
const qsa = (selector, ctx = document) => Array.from(ctx.querySelectorAll(selector));

const loadingEmojis = [
  "🥕", 
  "⏲️", 
  "🍲", 
  "🥦", 
  "🥘", 
  "🍲", 
  "🍽️", 
  "🌶️", 
  "🥗"
];
const loadingPhrases = [
  "Chopping the carrots",
  "Preheating the oven",
  "Simmering the sauce",
  "Baking the broccoli",
  "Whisking the batter",
  "Sautéing the veggies",
  "Plating the meal",
  "Toasting the spices",
  "Mixing the dressing",
];
let loadingIntervalId = null;
let loadingStep = 0;

const updateLoadingOverlay = () => {
  const overlay = qs("#loading-overlay");
  if (!overlay || overlay.hidden) {
    return;
  }
  const emojiEl = qs("#loading-emoji", overlay);
  const textEl = qs("#loading-text", overlay);
  const emoji = loadingEmojis[loadingStep % loadingEmojis.length];
  const phrase = loadingPhrases[loadingStep % loadingPhrases.length];
  if (emojiEl) {
    emojiEl.textContent = emoji;
  }
  if (textEl) {
    textEl.textContent = `${phrase}…`;
  }
  loadingStep += 1;
};

function showLoadingOverlay() {
  const overlay = qs("#loading-overlay");
  if (!overlay) return;

  if (loadingIntervalId) {
    clearInterval(loadingIntervalId);
  }

  loadingStep = Math.floor(Math.random() * loadingPhrases.length);
  overlay.hidden = false;
  overlay.setAttribute("aria-hidden", "false");
  updateLoadingOverlay();
  loadingIntervalId = window.setInterval(updateLoadingOverlay, 5000);
}

function hideLoadingOverlay() {
  if (loadingIntervalId) {
    clearInterval(loadingIntervalId);
    loadingIntervalId = null;
  }
  const overlay = qs("#loading-overlay");
  if (!overlay) return;
  overlay.hidden = true;
  overlay.setAttribute("aria-hidden", "true");
}

const formatMetricValue = (value, suffix = "") => {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const rounded = Math.round(value);
  return `${rounded}${suffix ? ` ${suffix}` : ""}`;
};

const getMetricStatus = (type, target, actual) => {
  if (!Number.isFinite(target) || !Number.isFinite(actual)) {
    return "neutral";
  }

  const safeTarget = target === 0 ? 0.0001 : target;
  const diff = actual - target;
  const absDiffRatio = Math.abs(diff) / Math.abs(safeTarget);

  switch (type) {
    case "calories":
      if (absDiffRatio <= 0.1) return "good";
      if (absDiffRatio <= 0.2) return "warn";
      return "bad";
    case "protein":
      if (actual >= target) return "good";
      if ((target - actual) / Math.abs(safeTarget) <= 0.1) return "warn";
      return "bad";
    case "carbs":
    case "fat":
      if (actual <= target) return "good";
      if ((actual - target) / Math.abs(safeTarget) <= 0.1) return "warn";
      return "bad";
    default:
      return "neutral";
  }
};

const buildDayMetrics = (day) => {
  const metricsConfig = [
    { type: "calories", label: "Calories", suffix: "cal", target: day?.target_calories, actual: day?.total_calories },
    { type: "protein", label: "Protein (min)", suffix: "g", target: day?.target_protein, actual: day?.total_protein },
    { type: "carbs", label: "Carbs (max)", suffix: "g", target: day?.target_carbs, actual: day?.total_carbs },
    { type: "fat", label: "Fat (max)", suffix: "g", target: day?.target_fat, actual: day?.total_fat },
  ];

  const container = document.createElement("div");
  container.className = "day-metrics";
  let hasContent = false;

  metricsConfig.forEach((metric) => {
    const hasValues = Number.isFinite(metric.target) || Number.isFinite(metric.actual);
    if (!hasValues) {
      return;
    }
    hasContent = true;

    const card = document.createElement("div");
    card.className = "day-metric";

    const labelEl = document.createElement("span");
    labelEl.className = "day-metric-label";
    labelEl.textContent = metric.label;
    card.appendChild(labelEl);

    const targetEl = document.createElement("span");
    targetEl.className = "day-metric-target";
    targetEl.textContent = `Target: ${formatMetricValue(metric.target, metric.suffix)}`;
    card.appendChild(targetEl);

    const actualEl = document.createElement("span");
    const status = getMetricStatus(metric.type, metric.target, metric.actual);
    actualEl.className = `day-metric-actual status-${status}`;
    actualEl.textContent = `Actual: ${formatMetricValue(metric.actual, metric.suffix)}`;
    card.appendChild(actualEl);

    card.dataset.metricType = metric.type;
    container.appendChild(card);
  });

  return hasContent ? container : null;
};

const formatMealTitle = (meal) => {
  const mealType = meal?.meal_type ? `${meal.meal_type}: ` : "";
  return `${mealType}${meal?.title ?? "Meal"}`;
};

const buildIngredients = (meal) => {
  const items = Array.isArray(meal?.ingredients) ? meal.ingredients : [];
  const quantities = Array.isArray(meal?.quantities) ? meal.quantities : [];
  const units = Array.isArray(meal?.units) ? meal.units : [];

  return items
    .map((ingredient, idx) => {
      const qty = quantities[idx] ?? "";
      const unit = units[idx] ?? "";
      return [qty, unit, ingredient].join(" ").trim();
    })
    .filter(Boolean);
};

const buildInstructions = (meal) => {
  if (!meal?.instructions) {
    return [];
  }

  return meal.instructions
    .split(/\r?\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
};

const toggleDayCard = (card, expand) => {
  const shouldExpand = typeof expand === "boolean" ? expand : !card.classList.contains("expanded");
  card.classList.toggle("expanded", shouldExpand);
  const btn = qs(".toggle-btn.day", card);
  if (btn) {
    btn.textContent = shouldExpand ? "Collapse day" : "Expand day";
    btn.setAttribute("aria-expanded", String(shouldExpand));
  }
};

const toggleMealCard = (card, expand) => {
  const shouldExpand = typeof expand === "boolean" ? expand : !card.classList.contains("expanded");
  card.classList.toggle("expanded", shouldExpand);
  const btn = qs(".toggle-btn.meal", card);
  if (btn) {
    btn.textContent = shouldExpand ? "Hide meal details" : "View meal details";
    btn.setAttribute("aria-expanded", String(shouldExpand));
  }
};

const renderMeal = (meal) => {
  if (!meal) {
    return null;
  }

  const card = document.createElement("article");
  card.className = "meal-card";

  const headline = document.createElement("div");
  headline.className = "meal-headline";

  const titleWrap = document.createElement("div");
  const title = document.createElement("h3");
  title.className = "meal-title";
  title.textContent = meal.title ?? "Meal idea";
  titleWrap.appendChild(title);

  if (meal.description) {
    const desc = document.createElement("p");
    desc.className = "meal-description";
    desc.textContent = meal.description;
    titleWrap.appendChild(desc);
  }

  const calories = document.createElement("div");
  calories.className = "meal-calories";
  calories.textContent = `${meal.calories ?? "—"} cal`;

  headline.appendChild(titleWrap);
  headline.appendChild(calories);

  const macros = document.createElement("div");
  macros.className = "macro-row";
  if (meal?.macros?.protein) {
    const p = document.createElement("span");
    p.textContent = `Protein: ${meal.macros.protein}`;
    macros.appendChild(p);
  }
  if (meal?.macros?.carbs) {
    const c = document.createElement("span");
    c.textContent = `Carbs: ${meal.macros.carbs}`;
    macros.appendChild(c);
  }
  if (meal?.macros?.fat) {
    const f = document.createElement("span");
    f.textContent = `Fat: ${meal.macros.fat}`;
    macros.appendChild(f);
  }

  const query = document.createElement("div");
  query.className = "meal-query";
  if (meal?.query) {
    const label = document.createElement("span");
    label.className = "meal-query-label";
    label.textContent = "Search query";

    const value = document.createElement("span");
    value.className = "meal-query-value";
    value.textContent = meal.query;

    query.appendChild(label);
    query.appendChild(value);
  } else {
    query.textContent = "Search query unavailable.";
    query.classList.add("is-muted");
  }

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "toggle-btn meal";
  toggle.textContent = "View meal details";
  toggle.setAttribute("aria-expanded", "false");

  const details = document.createElement("div");
  details.className = "meal-details";

  const ingredients = buildIngredients(meal);
  if (ingredients.length) {
    const section = document.createElement("section");
    section.className = "meal-ingredients";
    const heading = document.createElement("h4");
    heading.textContent = "Ingredients";
    const list = document.createElement("ul");
    ingredients.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item;
      list.appendChild(li);
    });
    section.appendChild(heading);
    section.appendChild(list);
    details.appendChild(section);
  }

  const instructions = buildInstructions(meal);
  if (instructions.length) {
    const section = document.createElement("section");
    section.className = "meal-instructions";
    const heading = document.createElement("h4");
    heading.textContent = "Instructions";
    section.appendChild(heading);
    instructions.forEach((line) => {
      const p = document.createElement("p");
      p.textContent = line;
      section.appendChild(p);
    });
    details.appendChild(section);
  }

  if (!ingredients.length && !instructions.length) {
    const p = document.createElement("p");
    p.textContent = "Detailed ingredients and instructions are not available for this meal.";
    details.appendChild(p);
  }

  toggle.addEventListener("click", () => toggleMealCard(card));

  card.appendChild(headline);
  card.appendChild(macros);
  card.appendChild(query);
  card.appendChild(toggle);
  card.appendChild(details);

  return card;
};

const renderDay = (day, idx) => {
  const article = document.createElement("article");
  article.className = "plan-day";

  const header = document.createElement("div");
  header.className = "day-header";

  const info = document.createElement("div");
  info.className = "day-info";

  const title = document.createElement("h2");
  const dayNumber = Number.isFinite(day?.day) ? day.day : idx + 1;
  title.textContent = `Day ${dayNumber}`;
  info.appendChild(title);

  const metrics = buildDayMetrics(day);
  if (metrics) {
    info.appendChild(metrics);
  }

  const preview = document.createElement("div");
  preview.className = "day-preview";
  (day?.meals ?? []).forEach((meal) => {
    if (!meal) return;
    const chip = document.createElement("span");
    chip.className = "preview-chip";
    chip.textContent = formatMealTitle(meal);
    preview.appendChild(chip);
  });
  info.appendChild(preview);

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "toggle-btn day";
  toggle.textContent = "Expand day";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => toggleDayCard(article));

  header.appendChild(info);
  header.appendChild(toggle);

  const detailWrap = document.createElement("div");
  detailWrap.className = "day-details";
  (day?.meals ?? []).forEach((meal) => {
    const mealNode = renderMeal(meal);
    if (mealNode) {
      detailWrap.appendChild(mealNode);
    }
  });

  article.appendChild(header);
  article.appendChild(detailWrap);

  return article;
};

const renderPlans = (dailyPlans) => {
  const resultsSection = qs("#results");
  const list = qs("#results-list");
  const hint = qs("#results-hint");

  list.innerHTML = "";

  if (!Array.isArray(dailyPlans) || !dailyPlans.length) {
    hint.textContent = "No plans returned. Try adjusting your inputs and generate again.";
    resultsSection.classList.remove("hidden");
    qs("#expand-all").disabled = true;
    qs("#collapse-all").disabled = true;
    return;
  }

  dailyPlans.forEach((day, idx) => {
    const node = renderDay(day, idx);
    list.appendChild(node);
  });

  hint.textContent = `Showing ${dailyPlans.length} day${dailyPlans.length > 1 ? "s" : ""} planned by the API.`;
  resultsSection.classList.remove("hidden");
  qs("#expand-all").disabled = false;
  qs("#collapse-all").disabled = false;
};

const setButtonLoading = (isLoading) => {
  const button = qs("#generate-btn");
  if (!button) return;

  if (isLoading) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Generating…";
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    button.appendChild(spinner);
    button.disabled = true;
    showLoadingOverlay();
  } else {
    const original = button.dataset.originalText || "Generate meal plan";
    button.textContent = original;
    button.disabled = false;
    hideLoadingOverlay();
  }
};

const showBanner = (text, variant = "error") => {
  let banner = qs(".banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.className = "banner";
    document.body.prepend(banner);
  }
  banner.textContent = text;
  banner.dataset.variant = variant;
  banner.hidden = false;
  setTimeout(() => {
    banner.hidden = true;
  }, 4000);
};

const expandAll = (expand) => {
  qsa(".plan-day").forEach((card) => toggleDayCard(card, expand));
  qsa(".plan-day .meal-card").forEach((card) => toggleMealCard(card, expand));
};

const wireHandlers = () => {
  const form = qs("#planner-form");
  const expandBtn = qs("#expand-all");
  const collapseBtn = qs("#collapse-all");

  expandBtn.addEventListener("click", () => expandAll(true));
  collapseBtn.addEventListener("click", () => expandAll(false));

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);
    const errors = [];
    const preferences = (formData.get("preferences") || "").toString().trim();
    const exclusions = (formData.get("exclusions") || "").toString().trim();
    const payload = {
      target_calories: Number(formData.get("target_calories")),
      limit_per_meal: Number(5),
      num_days: Number(formData.get("num_days")),
      dietary: formData.getAll("dietary").filter(Boolean),
    };

    if (!payload.target_calories || payload.target_calories <= 0) {
      showBanner("Please provide a valid target calorie amount.", "error");
      return;
    }

    if (preferences) {
      payload.preferences = preferences;
    }

    if (exclusions) {
      payload.exclusions = exclusions;
    }

    [
      ["target_protein", "target protein"],
      ["target_fat", "target fat"],
      ["target_carbs", "target carbs"],
    ].forEach(([field, label]) => {
      const rawValue = formData.get(field);
      const value = typeof rawValue === "string" ? rawValue.trim() : "";
      if (!value) {
        return;
      }
      const parsed = Number(value);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        errors.push(`Please provide a positive number for ${label}.`);
        return;
      }
      payload[field] = parsed;
    });

    console.log(payload);

    if (errors.length) {
      showBanner(errors[0], "error");
      return;
    }

    setButtonLoading(true);
    try {
      const response = await fetch("/meal-planning/n-day", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        const message =
          error?.detail?.message ||
          error?.detail ||
          error?.error ||
          `Request failed with status ${response.status}`;
        throw new Error(message);
      }

      const data = await response.json();
      renderPlans(data?.daily_plans);
      expandAll(false);
      showBanner("Meal plan generated successfully!", "success");
    } catch (error) {
      console.error(error);
      showBanner(error.message || "Unable to generate a meal plan right now.", "error");
    } finally {
      setButtonLoading(false);
    }
  });
};

document.addEventListener("DOMContentLoaded", () => {
  wireHandlers();
});
