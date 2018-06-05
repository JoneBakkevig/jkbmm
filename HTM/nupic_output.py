# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
"""
Provides two classes with the same signature for writing data out of NuPIC
models.
(This is a component of the One Hot Gym Prediction Tutorial.)
"""
import csv
from collections import deque
from abc import ABCMeta, abstractmethod
# Try to import matplotlib, but we don't have to.

import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num


WINDOW = 100


class NuPICOutput(object):
    __metaclass__ = ABCMeta

    def __init__(self, names, show_anomaly_score=False):
        self.names = names
        self.show_anomaly_score = show_anomaly_score

    @abstractmethod
    def write(self, timestamps, actual_values, predicted_values,
              prediction_step=1):
        pass

    @abstractmethod
    def close(self):
        pass


class NuPICFileOutput(NuPICOutput):

    def __init__(self, *args, **kwargs):
        super(NuPICFileOutput, self).__init__(*args, **kwargs)
        self.output_files = []
        self.output_writers = []
        self.line_counts = []
        header_row = ['date', 'amount', 'prediction']
        for name in self.names:
            self.line_counts.append(0)
            output_file_name = "%s_out.csv" % name
            print "Preparing to output %s data to %s" % (name, output_file_name)
            output_file = open(output_file_name, "w")
            self.output_files.append(output_file)
            output_writer = csv.writer(output_file)
            self.output_files.append(output_writer)
            output_writer.writerow(header_row)

    def write(self, timestamps, actual_values, predicted_values,
              prediction_step=1):

        assert len(timestamps) == len(actual_values) == len(predicted_values)

        for index in range(len(self.names)):
            timestamp = timestamps[index]
            actual = actual_values[index]
            prediction = predicted_values[index]
            writer = self.output_writers[index]

            if timestamp is not None:
                output_row = [timestamp, actual, prediction]
                writer.writerow(output_row)
                self.line_counts[index] += 1

    def close(self):
        for index, name in enumerate(self.names):
            self.output_files[index].close()
            print "Done. Wrote %i data lines to %s." % (self.line_counts[index], name)


class NuPICPlotOutput(NuPICOutput):

    def __init__(self, *args, **kwargs):
        super(NuPICPlotOutput, self).__init__(*args, **kwargs)
        # Turn matplotlib interactive mode on.
        plt.ion()
        self.dates = []
        self.converted_dates = []
        self.actual_values = []
        self.predicted_values = []
        self.actual_lines = []
        self.predicted_lines = []
        self.lines_initialized = False
        self.graphs = []
        plot_count = len(self.names)
        plot_height = max(plot_count * 3, 6)
        fig = plt.figure(figsize=(14, plot_height))
        gs = gridspec.GridSpec(plot_count, 1)
        for index in range(len(self.names)):
            self.graphs.append(fig.add_subplot(gs[index, 0]))
            plt.title("")
            plt.ylabel('Absolute TTE error in days')
            plt.xlabel('Date')
        plt.tight_layout()

    def initialize_lines(self, timestamps):
        for index in range(len(self.names)):
            print "initializing %s" % self.names[index]
            # graph = self.graphs[index]
            self.dates.append(deque([timestamps[index]] * WINDOW, maxlen=WINDOW))
            self.converted_dates.append(deque(
                [date2num(date) for date in self.dates[index]], maxlen=WINDOW
            ))
            self.actual_values.append(deque([0.0] * WINDOW, maxlen=WINDOW))
            self.predicted_values.append(deque([0.0] * WINDOW, maxlen=WINDOW))

            actualPlot, = self.graphs[index].plot(
                self.dates[index], self.actual_values[index]
            )
            self.actual_lines.append(actualPlot)
            predicted_plot, = self.graphs[index].plot(
                self.dates[index], self.predicted_values[index]
            )
            self.predicted_lines.append(predicted_plot)
        self.lines_initialized = True

    def write(self, timestamps, actual_values, predicted_values,
              prediction_step=1):

        assert len(timestamps) == len(actual_values) == len(predicted_values)

        # We need the first timestamp to initialize the lines at the right X value,
        # so do that check first.
        if not self.lines_initialized:
            self.initialize_lines(timestamps)

        for index in range(len(self.names)):
            self.dates[index].append(timestamps[index])
            self.converted_dates[index].append(date2num(timestamps[index]))
            error = 0
            try:
                error = abs(actual_values[index] - predicted_values[index])
            except TypeError:
                pass

            self.actual_values[index].append(error)
            self.predicted_values[index].append(error)

            #self.actual_values[index].append(actual_values[index])
            #self.predicted_values[index].append(predicted_values[index])

            # Update data
            self.actual_lines[index].set_xdata(self.converted_dates[index])
            self.actual_lines[index].set_ydata(self.actual_values[index])
            self.actual_lines[index].set_linestyle("-")
            self.actual_lines[index].set_color("grey")
            self.predicted_lines[index].set_xdata(self.converted_dates[index])
            self.predicted_lines[index].set_ydata(self.predicted_values[index])
            self.predicted_lines[index].set_linestyle(":")
            self.predicted_lines[index].set_color("grey")

            self.graphs[index].relim()
            self.graphs[index].autoscale_view(True, True, True)

        plt.draw()
        #plt.legend(('Actual', 'Predicted'), loc=3)

    def refresh_gui(self):
        """
        Give plot a pause, so data is drawn and GUI's event loop can run.
        """
        plt.pause(0.1001)

    def close(self):
        plt.ioff()
        plt.show()


NuPICOutput.register(NuPICFileOutput)
NuPICOutput.register(NuPICPlotOutput)
