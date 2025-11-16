from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import threading
import random
import time
from typing import List, Dict
from pydantic import BaseModel

app = FastAPI()

# Configuration
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
NUM_ENEMIES = 5
NUM_ALLIES = 5
UPDATE_INTERVAL = 0.25  # seconds

# State
enemies: List[Dict[str, float]] = []
allies: List[Dict[str, float]] = []
running = False


class Position(BaseModel):
    x: float
    y: float


class PositionsResponse(BaseModel):
    enemies: List[Position]
    allies: List[Position]

def get_distance(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """Calculates Euclidean distance between two positions"""
    return ((pos1['x'] - pos2['x'])**2 + (pos1['y'] - pos2['y'])**2)**0.5


def random_walk(position: Dict[str, float], step_size: float = 5.0):
    """Update position with random walk, keeping within bounds"""
    position['x'] += random.uniform(-step_size, step_size)
    position['y'] += random.uniform(-step_size, step_size)

    # Keep within canvas bounds
    position['x'] = max(0, min(CANVAS_WIDTH, position['x']))
    position['y'] = max(0, min(CANVAS_HEIGHT, position['y']))


def simulation_loop():
    """Background thread that continuously updates positions"""
    global running, enemies, allies
    running = True
    ENVIRONMENT_NOISE = 0.0005  # A "quiet" ocean. Try 0.005 for a "loud" one.
    ALPHA = 100

    while running:
        # Update enemies
        for enemy in enemies:
            random_walk(enemy)
        
        # Update allies
        for ally in allies:
            random_walk(ally)


        for sender in allies:
            for receiver in allies:
                if sender['id'] == receiver['id']:
                    continue  # A drone can't talk to itself

                distance = get_distance(sender, receiver)
                
                # Signal strength = 1 / distance^2 (inverse-square law)
                # We add 1 to distance to avoid division by zero if they are on top of each other
                signal_strength = ALPHA * (1 / (distance**2 + 1))
                
                # --- THIS IS YOUR CORE HYPOTHESIS TEST ---
                if signal_strength > ENVIRONMENT_NOISE:
                    # SUCCESS
                    print(f"Tick {int(time.time())}: Ally {sender['id']} -> Ally {receiver['id']} | SUCCESS (Dist: {distance:.1f}, Signal: {signal_strength:.6f})")
                else:
                    # FAILURE
                    print(f"Tick {int(time.time())}: Ally {sender['id']} -> Ally {receiver['id']} | FAILED (Dist: {distance:.1f}, Signal: {signal_strength:.6f})")
        
        time.sleep(UPDATE_INTERVAL)


def initialize_objects():
    """Initialize enemies and allies with random starting positions"""
    global enemies, allies
    
    enemies = [
        {'id': i, 'x': random.uniform(0, CANVAS_WIDTH), 'y': random.uniform(0, CANVAS_HEIGHT)}
        for i in range(NUM_ENEMIES)
    ]
    
    allies = [
        {'id': i, 'x': random.uniform(0, CANVAS_WIDTH), 'y': random.uniform(0, CANVAS_HEIGHT)}
        for i in range(NUM_ALLIES)
    ]


@app.on_event("startup")
async def startup_event():
    """Initialize objects and start simulation thread"""
    initialize_objects()
    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop simulation loop"""
    global running
    running = False


@app.get("/api/positions")
async def get_positions() -> PositionsResponse:
    """Return current positions of all enemies and allies"""
    return PositionsResponse(
        enemies=[Position(x=e['x'], y=e['y']) for e in enemies],
        allies=[Position(x=a['x'], y=a['y']) for a in allies]
    )


@app.get("/")
async def read_root():
    """Serve the HTML file"""
    return FileResponse('index.html')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6969)

