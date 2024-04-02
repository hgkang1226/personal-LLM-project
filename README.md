# Personal LLM Project
This project is a personal LLM (Large Language Model) project. It consists of a frontend, backend and practice code of training and finetuning LLaMA.

## Frontend
The frontend is implemented using Streamlit. Use the following command to run the frontend:
```bash
streamlit run service/frontend.py --server.address 0.0.0.0
```
This command runs the Streamlit server and allows access from all network interfaces.

## Backend
### Developer Mode
In developer mode, you can use Uvicorn to run the FastAPI application. Use the following command:
```bash
uvicorn backend:app --reload
```
This command automatically restarts the server whenever code changes are detected.

### Production Mode
In production mode, it is recommended to use Gunicorn to run the FastAPI application. Use the following command:
```bash
gunicorn backend:app -w 4 -k uvicorn.workers.UvicornWorker
```
This command:

* ```-w``` 4: Uses 4 worker processes.
* ```-k``` uvicorn.workers.UvicornWorker: Uses Uvicorn workers to support asynchronous execution.
You can adjust the number of workers (-w) according to your needs.