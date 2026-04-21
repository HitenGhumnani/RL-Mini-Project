const MIN_PRICE = 10;
const MAX_PRICE = 100;
const PRICE_STEP = 5;
const TOTAL_STEPS = 30;

const BASE_DEMAND = 100;
const PRICE_SENSITIVITY = 2;
const NOISE_MIN = -5;
const NOISE_MAX = 5;

const initialPriceInput = document.getElementById("initialPrice");
const startBtn = document.getElementById("startBtn");
const resultsBody = document.getElementById("resultsBody");
const statusText = document.getElementById("statusText");
const chartCanvas = document.getElementById("priceChart");

const dataService = {
  async simulate(initialPrice, steps) {
    const response = await fetch("http://127.0.0.1:5000/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        initial_price: initialPrice,
        steps
      })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || "Backend request failed");
    }

    const data = await response.json();
    return data.rows;
  }
};

startBtn.addEventListener("click", runSimulation);

drawChart([]);

async function runSimulation() {
  const initialPrice = Number(initialPriceInput.value);

  if (!isValidPrice(initialPrice)) {
    statusText.textContent = `Please enter a price between ${MIN_PRICE} and ${MAX_PRICE} in steps of ${PRICE_STEP}.`;
    return;
  }

  startBtn.disabled = true;
  statusText.textContent = "Simulating...";

  try {
    const rows = await dataService.simulate(initialPrice, TOTAL_STEPS);
    renderTable(rows);
    drawChart(rows.map((row) => row.price));
    statusText.textContent = `Simulation complete: ${rows.length} steps.`;
  } catch (error) {
    statusText.textContent = "Simulation failed. Check console for details.";
    console.error(error);
  } finally {
    startBtn.disabled = false;
  }
}

function simulateLocally(initialPrice, steps) {
  const rows = [];
  let price = snapToStep(initialPrice);

  for (let i = 1; i <= steps; i += 1) {
    const demand = getDemand(price);

    let action = 0;
    if (demand > 45) {
      action = 1;
    } else if (demand < 25) {
      action = -1;
    }

    rows.push({
      step: i,
      price,
      demand,
      action
    });

    price = clampPrice(price + action * PRICE_STEP);
  }

  return rows;
}

function getDemand(price) {
  const noise = randomInt(NOISE_MIN, NOISE_MAX);
  return Math.max(0, BASE_DEMAND - PRICE_SENSITIVITY * price + noise);
}

function renderTable(rows) {
  resultsBody.innerHTML = "";

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.step}</td>
      <td>${row.price}</td>
      <td>${row.demand}</td>
      <td>${formatAction(row.action)}</td>
    `;
    resultsBody.appendChild(tr);
  });
}

function drawChart(prices) {
  const ctx = chartCanvas.getContext("2d");
  const width = chartCanvas.width;
  const height = chartCanvas.height;

  ctx.clearRect(0, 0, width, height);

  drawAxes(ctx, width, height);

  if (!prices.length) {
    return;
  }

  const padding = 36;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;

  ctx.beginPath();
  prices.forEach((price, index) => {
    const x = padding + (index / (prices.length - 1 || 1)) * graphWidth;
    const y = padding + ((MAX_PRICE - price) / (MAX_PRICE - MIN_PRICE)) * graphHeight;

    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.strokeStyle = "#1f7a8c";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawAxes(ctx, width, height) {
  const padding = 36;

  ctx.strokeStyle = "#9db0c4";
  ctx.lineWidth = 1;

  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, height - padding);
  ctx.lineTo(width - padding, height - padding);
  ctx.stroke();

  ctx.fillStyle = "#5c6b7a";
  ctx.font = "12px Space Grotesk";
  ctx.fillText("Price", 8, 20);
  ctx.fillText("Step", width - 50, height - 10);
}

function isValidPrice(value) {
  return Number.isFinite(value) && value >= MIN_PRICE && value <= MAX_PRICE && (value - MIN_PRICE) % PRICE_STEP === 0;
}

function clampPrice(price) {
  return Math.max(MIN_PRICE, Math.min(MAX_PRICE, price));
}

function snapToStep(price) {
  return MIN_PRICE + Math.round((price - MIN_PRICE) / PRICE_STEP) * PRICE_STEP;
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function formatAction(action) {
  if (action > 0) {
    return "+1";
  }

  if (action < 0) {
    return "-1";
  }

  return "0";
}
