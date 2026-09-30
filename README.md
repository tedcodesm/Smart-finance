#  Personal Finance Application

  modern, comprehensive personal finance application designed to help you track transactions, manage budgets, set savings goals, and leverage machine learning to predict your future financial position.

## Features

- **Transactions**: Log income and expenses by category.
- **Budgets**: Set spending limits and receive alerts when you approach them.
- **Goals**: Set long-term savings goals and track your progress based on net cash flow.
- **Predictions**: Nightly ML batch jobs analyze your spending habits to forecast your future financial health.
- **Reports**: Beautiful, responsive charts to help you visualize your data.

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


