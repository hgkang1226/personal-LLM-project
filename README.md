# Frontend
```streamlit run service/frontend.py --server.address 0.0.0.0```
# Backend
## Developer
```uvicorn backend:app --reload```
## Production
```gunicorn backend:app -w 4 -k uvicorn.workers.UvicornWorker```