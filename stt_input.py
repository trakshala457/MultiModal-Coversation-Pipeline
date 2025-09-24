# We are initializing the ASR model here so it's ready to use when the server starts.
# Loading the model takes time, so we do it once at startup.
model = whisper.load_model("base")

app = FastAPI()

# This is our ASR endpoint




    

