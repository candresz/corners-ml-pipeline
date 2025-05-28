from prefect import flow

@flow
def corners_live_pipeline():
    import src.train_model  

if __name__ == "__main__":
    corners_live_pipeline()
