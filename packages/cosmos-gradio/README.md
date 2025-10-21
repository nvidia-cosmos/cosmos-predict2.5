
# Model Deployment with Cosmos-Gradio

This document describes how to add a preconfigured Gradio interface to your model for easy interaction.

UI features:
* generic inference parameter input field in json format
* output video and image viewer
* file uploader for image and video assets
* log file viewer


gradio app features:
* starts model worker processes
* dispatches input requests in json format to workers
* collects output files and status json from workers

## Preparing a model to use Cosmos-Gradio

A model needs to provide the following components to use the cosmos_gradio components:

1. A model implementation that can be launched by the gradio implementation as a worker process.
   * The class should load all required models in the initializer.
   * The class must have an infer function that takes the inference arguments as a dictionary.
   * The args dictionary needs to be serializable with json.dumps().
   * Generated outputs need to be saved in the folder specified in the output_dir argument.

```python
class MyModel:
   def __init__(**kwargs):
      self.pipeline = MyPipeline(**kwargs)

   def infer(self, args: dict):
      # call underlying model
      outputs = self.pipeline(**args)

      # save outputs in 'output_dir'
```

2. A factory function needs to be provdied that handles model specific initialization. The gradio implementation will call this function to create the model in each worker process.
```python
def create_worker():
    # configure pipeline as needed using environment variables
    model = MyModel(os.getenv("CHECKPOINT_NAME"))
    return model
```


3. The default UI in cosmos-gradio creates a parameter dictionary that is parsed out of the JSON input field.
   A parameter validation function running in the UI process will keep the UI responsive:
* Invalid input is very common as users have to edit JSON files.
* Downstream validation through model in worker process requires a roundtrip through model input queue and model.
* Missing validation altogether can lead to unreadable exceptions.

```python

def validate_my_parameters(params: dict) -> dict:
   """Validate inference parameters before sending to workers."""
   # Add your parameter validation logic here
   # raise exception with explanation of parameter error

   # fix up invalid parameters as needed and return a valid parameter set
   return params
```


4. Create a model specific bootstrapper:

The bootstrapper puts it all together:
* Create and configure the distributed model.
* Create and configure your Gradio interface.
* Launch the server.

The GradioApp launches worker processes and dispatches incoming requests in its infer function:
```python

app = GradioApp(
    validator=validate_my_parameters,
    factory_module="my_package.my_model"
    factory_function="create_worker",
)
```

Alternatively, create your own Gradio app if you already have a distributed model that can dispatch incoming inference requests.


Create the default UI with:

```python
interface = create_gradio_UI(infer_func=app.infer, header="my generative model")
```

Alternatively, create your own UI that delivers a JSON-based dictionary of inference arguments.

Finally, launch the gradio interface on the desired port:
```python
interface.launch(server_name="0.0.0.0", server_port=8080)
```

## Running the server

The Cosmos-Gradio is run in your models environment as a dependency. E.g. if your model requires a docker to run your model:

1. **Run the Docker container:**
   ```bash
   # Run the container with GPU support, port mapping, and volume mounts
   docker run --gpus all -it --rm \
     -p 8080:8080 \
     -v /path/to/cosmos-predict2:/workspace \
     -v /path/to/checkpoints:/workspace/checkpoints \
     -v /path/to/datasets:/workspace/datasets \
     <your docker>
   ```
2. **Inside the container, setup Gradio:**
   ```bash
   pip install <path to mounted cosmos-gradio source>
   ```

3. **Inside the container, launch the Gradio server:**
   ```bash
   # Set up the environment needed for your model
   export PYTHONPATH=/workspace
   cd /workspace
   export NUM_GPUS=1
   export MODEL_NAME="my_model"
   export CHECKPOINT_DIR="/workspace/checkpoints"
   # the log file viewer will read from $LOG_FILE
   export LOG_FILE="gradio.log"

   # Launch the Gradio server
   python3 sample/bootstrapper.py 2>&1 | tee -a $LOG_FILE
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://localhost:8080`

## Deployment Configuration

The DeploymentEnv class provides environment variables to configure the Gradio UI and the model IPC:
* MODEL_NAME is used to create model-specific file names. This environment variable should also be used to pick model-specific configuration in your bootstrapper.
* NUM_GPUS specifies how many worker processes should be spawned.
* UPLOADS_DIR
* OUTPUT_DIR
* LOG_FILE

Additional environment variables are predefined that should be used for model configuration:
* CHECKPOINT_DIR
