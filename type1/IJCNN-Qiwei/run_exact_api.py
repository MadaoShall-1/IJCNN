#!/usr/bin/env python3
from ijcnn_qiwei.exact_api import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
