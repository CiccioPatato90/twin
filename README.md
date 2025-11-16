# Random Walk Simulation

A simple prototype that simulates enemies and allies moving randomly on a 2D plane.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
```

2. Activate the virtual environment:
```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

Start the server:
```bash
python server.py
```

Then open your browser and navigate to:
```
http://localhost:8000
```

## Configuration

You can modify the number of enemies and allies in `server.py`:
- `NUM_ENEMIES`: Number of enemy objects (default: 5)
- `NUM_ALLIES`: Number of ally objects (default: 5)
- `CANVAS_WIDTH` and `CANVAS_HEIGHT`: Canvas dimensions (default: 800x600)
- `UPDATE_INTERVAL`: Simulation update frequency in seconds (default: 0.1)


## API CONTRACT

{
  enemies: [{x: number, y: number}, ...],
  allies: [{x: number, y: number}, ...]
}

As long as you do not break this, the client html will continue working