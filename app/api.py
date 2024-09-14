import lib
import fastapi
import pydantic as p
from typing import Annotated, Literal
import uvicorn
import pandas as pd
import asyncio

app = fastapi.FastAPI()


@app.post("/predict/{item_id}")
async def predict(
        item_id: str,
        main_file: Annotated[fastapi.UploadFile, fastapi.File(description="A file read as UploadFile")],
        sales_prices_file: Annotated[fastapi.UploadFile, fastapi.File(description="A file read as UploadFile")],
        sales_date_file: Annotated[fastapi.UploadFile, fastapi.File(description="A file read as UploadFile")],
        granularity: Literal['Day', 'Week', 'Month'],
        horizont: Annotated[int, p.Field(int, ge=1, le=10)]
):
    dataset: pd.DataFrame = lib.merge_files_to_dataset(
        pd.read_csv(main_file.file),
        pd.read_csv(sales_prices_file.file),
        pd.read_csv(sales_date_file.file)
    )
    dataset = lib.summ_sales_data(dataset, 'date', granularity=granularity)
    dataset = lib.df_encoding(dataset)

    data_prediction, _ = await asyncio.to_thread(lib.get_preds, dataset, [item_id], horizont)
    return data_prediction.to_json()

uvicorn.run(app, host='0.0.0.0', port=50001, log_level="info")
