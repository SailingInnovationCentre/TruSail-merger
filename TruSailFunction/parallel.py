import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def parallel_plot(Cars,name_boats):


	df = DataFrame(Cars, columns=['Brand', 'Rudder','Heel','Pitch'])

	df['Brand'] = pd.cut(df['Brand'], [0, 1, 2, 3,4,5])

	cols = ['Rudder','Heel','Pitch']
	x = [i for i, _ in enumerate(cols)]
	colours2 = ['#0000ff','#ff0000', '#008000','#000000','#000000']

	# create dict of categories: colours
	colours = {df['Brand'].cat.categories[i]: colours2[i] for i, _ in enumerate(df['Brand'].cat.categories)}

	# Create (X-1) sublots along x axis
	fig2, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(7.5, 8))
	canvas = FigureCanvas(fig2)
	# Get min, max and range for each column
	# Normalize the data for each column
	min_max_range = {}

	for col in cols:

		min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
		#min_max_range[col] = [-5.0, 5.0, 0.0]
		df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

	#print('Min max: ',min_max_range)

	# Plot each rowprint(num_files_directory)
	for i, ax_para in enumerate(axes):
		for idx in df.index:
			mpg_category = df.loc[idx, 'Brand']
			#print(idx, mpg_category,colours2[idx])
			ax_para.plot(x, df.loc[idx, cols], c=colours2[idx])
		ax_para.set_xlim([x[i], x[i + 1]])
		#ax_para.set_xlim([-5.0,5.0])

	# Set the tick positions and labels on y axis for each plot
	# Tick positions based on normalised data
	# Tick labels are based on original data
	def set_ticks_for_axis(dim, ax, ticks):

		min_val, max_val, val_range = min_max_range[cols[dim]]
		step = val_range / float(ticks - 1)
		tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
		norm_min = df[cols[dim]].min()
		norm_range = np.ptp(df[cols[dim]])
		norm_step = norm_range / float(ticks - 1)
		ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
		ax.yaxis.set_ticks(ticks)
		ax.set_yticklabels(tick_labels)

	for dim, ax_para in enumerate(axes):
		ax_para.xaxis.set_major_locator(ticker.FixedLocator([dim]))
		set_ticks_for_axis(dim, ax_para, ticks=6)
		ax_para.set_xticklabels([cols[dim]])

	# Move the final axis' ticks to the right-hand side
	ax_para = plt.twinx(axes[-1])
	dim = len(axes)
	ax_para.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
	set_ticks_for_axis(dim, ax_para, ticks=6)
	ax_para.set_xticklabels([cols[-2], cols[-1]])

	# Remove space between subplots
	plt.subplots_adjust(wspace=0)
	#fig2.savefig('results/2Bootje_' + name_boats[11:18] + '.png')
	fig2.canvas.draw()
	image_from_plot = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
	image_from_plot = image_from_plot.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

	return image_from_plot

