#  Personal Finance Application

  modern, comprehensive personal finance application designed to help you track transactions, manage budgets, set savings goals, and leverage machine learning to predict your future financial position.

## Features

### 1. Transactions
This is the foundational data of the app. Every time you earn money (income) or spend money (expense), it is logged here.

- **What it does:** It tracks the date, amount, description, and category (e.g., Groceries, Rent, Salary).
- **Why it matters:** Everything else in the application (Budgets, Goals, Predictions) relies on reading your transaction history to figure out your financial health.

### 2. Budgets
Budgets act as limits you set on your spending for specific categories over a specific time period (usually a month).

- **What it does:** You might say, "I only want to spend $400 on Groceries this month." The app actively compares your actual Transactions tagged as "Groceries" against this $400 limit.
- **Alerts:** If you hit 80% of your limit (based on the `BUDGET_ALERT_THRESHOLD_PCT=0.8` setting in your `.env`), the app will generate an alert to warn you that you are close to overspending.

### 3. Goals (Savings Goals)
Goals are long-term targets you want to hit, like saving for a "Vacation" or an "Emergency Fund."

- **What it does:** You set a target amount and a target date. As you have leftover income (Net Cash Flow) at the end of each month, the app tracks your progress toward hitting those targets. It helps you visualize how far along you are and if you are on track to meet your deadline.

### 4. Predictions (Machine Learning)
This is where the app gets smart.

- **What it does:** Instead of just looking at the past, the backend has a machine learning (ML) scheduler that wakes up at 2:17 AM to analyze your transaction history.
- **Why it matters:** It learns your spending habits and recurring bills to predict how much money you will likely spend next month, or whether you will have enough cash flow to hit your Savings Goals in time. It automatically forecasts your future financial position.

### 5. Reports
Reports take all of the raw data from your Transactions and turn it into actionable insights.

- **What it does:** It groups your data into beautiful charts and summaries. Instead of looking at 50 individual coffee purchases, a report aggregates it so you can see "I spent 15% of my total income on Dining Out this quarter, which is 5% more than last quarter."

---

## Getting Started

### Prerequisites
- **Node.js** (v18+)
- **Python** (3.10+)
- **MongoDB** (running locally on port 27017, or via Docker)

---

### 1. Database Setup

Ensure you have a MongoDB instance running locally. If you use Docker, you can start one with:
```bash
docker run -d -p 27017:27017 --name mercy-mongo mongo:latest
```

### 2. Backend Setup (FastAPI + Python)

The backend is built with FastAPI and utilizes a machine learning scheduler.

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Copy the example environment file if you haven't already:
   ```bash
   cp .env.example .env
   ```
   *Note: If `IN_MEMORY_DB=true` is set in the `.env`, the app will use an ephemeral mock database. Change it to `false` to use your real MongoDB instance.*
5. **Start the backend server:**
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```

### 3. Frontend Setup (Vite + React)

The frontend is a React SPA powered by Vite, featuring a premium glassmorphism UI.

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```
2. **Install dependencies:**
   ```bash
   npm install
   ```
3. **Start the development server:**
   ```bash
   npm run dev
   ```

### 4. Access the Application

Open your browser and navigate to **http://localhost:5173**. 


