from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import pandas as pd
import pickle
from typing import Optional

app = FastAPI()


class CodeExecutionRequest(BaseModel):
    code: str
    label: Optional[int] = 0


@app.post("/execute/")
async def execute_code(request: CodeExecutionRequest):
    # Create a DataFrame from the provided code
    df = pd.DataFrame(
        {
            "label": [request.label],
            "code": [request.code],
        }
    )

    # Save DataFrame to a pickle file
    try:
        with open("./dataset/trvd_test.pkl", "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    responses = {}

    # Run normalization script
    try:
        process = subprocess.run(
            ["python", "./normalization.py"], capture_output=True, text=True, check=True
        )
        responses["normalization"] = process.stdout
    except subprocess.CalledProcessError as e:
        return {
            "error": "Normalization script failed",
            "details": str(e),
            "output": e.stdout,
        }

    # Run pipeline script
    try:
        process = subprocess.run(
            ["python", "./pipeline.py"], capture_output=True, text=True, check=True
        )
        responses["pipeline"] = process.stdout
    except subprocess.CalledProcessError as e:
        return {
            "error": "Pipeline script failed",
            "details": str(e),
            "output": e.stdout,
        }

    # Run evaluation script
    try:
        process = subprocess.run(
            ["python", "./evaluation.py"], capture_output=True, text=True, check=True
        )
        responses["evaluation"] = process.stdout
    except subprocess.CalledProcessError as e:
        return {
            "error": "Evaluation script failed",
            "details": str(e),
            "output": e.stdout,
        }

    return {"message": "Execution completed successfully", "results": responses}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
