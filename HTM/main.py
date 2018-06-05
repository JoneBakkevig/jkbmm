import importlib
import sys
import csv
import datetime

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.frameworks.opf.prediction_metrics_manager import MetricsManager

import nupic_output

DESCRIPTION = (
  "Starts a NuPIC model from the model params returned by the swarm\n"
  "and pushes each line of input from the gym into the model. Results\n"
  "are written to an output file (default) or plotted dynamically if\n"
  "the --plot option is specified.\n"
  "NOTE: You must run ./swarm.py before this, because model parameters\n"
  "are required to run NuPIC.\n"
)

FILE_NAME = "csv_file"
DATA_DIR = "data/"
MODEL_PARAMS_DIR = "./model_0"
DATE_FORMAT = "%Y-%m-%d"

_METRIC_SPECS = (
    MetricSpec(field='tte', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'rmse', 'window': 1000, 'steps': 1}),
    MetricSpec(field='tte', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'rmse', 'window': 1000, 'steps': 1}),
    MetricSpec(field='tte', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'rmse', 'window': 1000, 'steps': 1}),
    MetricSpec(field='tte', metric='trivial',
               inferenceElement='prediction',
               params={'errorMetric': 'rmse', 'window': 1000, 'steps': 1}),
)


def create_model(model_params):
    model = ModelFactory.create(model_params)
    model.enableInference({"predictedField": "tte"})
    return model


def get_model_params_from_name(file_name):
    import_name = "model_0.model_params"
    print "Importing model params from %s" % import_name
    try:
        imported_model_params = importlib.import_module(import_name).MODEL_PARAMS
    except ImportError:
        raise Exception("No model params exist for '%s'. Run swarm first!"
                        % file_name)
    return imported_model_params


def run_io_through_nupic(input_data, model, file_name, plot, print_results):
    input_file = open(input_data, "rb")
    csv_reader = csv.reader(input_file)
    # skip header rows
    csv_reader.next()
    csv_reader.next()
    csv_reader.next()

    shifter = InferenceShifter()
    if plot:
        output = nupic_output.NuPICPlotOutput([file_name])
    else:
        output = nupic_output.NuPICFileOutput([file_name])

    metrics_manager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                     model.getInferenceType())

    counter = 0
    timestamp = None
    consumption = None

    result = None
    for row in csv_reader:
        counter += 1
        timestamp = datetime.datetime.strptime(row[1], DATE_FORMAT)
        consumption = int(row[2])
        amount = float(row[0])
        result = model.run({
            "amount": amount,
            "date": timestamp,
            "tte": consumption
        })
        result.metrics = metrics_manager.update(result)

        if counter % 100 == 0 or counter % 384 == 0:
            print "Read %i lines..." % counter
            print ("After %i records, rmse=%f" % (counter,
                                                            result.metrics["multiStepBestPredictions:multiStep:"
                                                                           "errorMetric='rmse':steps=1:window=1000:"
                                                                           "field=tte"]))

        if plot:
            result = shifter.shift(result)

        prediction = result.inferences["multiStepBestPredictions"][1]
        output.write([timestamp], [consumption], [prediction])
        if print_results:
            print("date:", timestamp.strftime("%y-%m-%d"), "actual:", consumption, "predicted:", prediction)

        if plot and counter % 20 == 0:
            output.refresh_gui()

        #if plot and counter % 1000 == 0:
        #    break

    input_file.close()
    output.close()


def run_model(file_name, plot=False, print_results=False):
    print "Creating model from %s..." % file_name
    model = create_model(get_model_params_from_name(file_name))
    input_data = "%s/%s.csv" % (DATA_DIR, file_name.replace(" ", "_"))
    run_io_through_nupic(input_data, model, file_name, plot, print_results)


if __name__ == "__main__":
    print DESCRIPTION
    plot = False
    print_results = False
    args = sys.argv[1:]
    if "--plot" in args:
        plot = True
    if "--print" in args:
        print_results = True
    run_model(FILE_NAME, plot=plot, print_results=print_results)
