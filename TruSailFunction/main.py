"""
Code developed by Marcos Pieras ( m.pierassagardoy@tudelft.nl )
Version 17 Nov 19
"""
from pandas import DataFrame
import pandas as pd
import numpy as np
import sys
import os
from parallel import *
import cv2
from math import pi
from os import listdir
from os.path import isfile, join

def compute_average(data_set,name_variable):
	return data_set[name_variable].mean()

def compute_travel(data_set,name_variable):
	denominator = data_set[name_variable].count()
	denominator.astype(float)

	return data_set[name_variable].diff().abs().sum()*60.0/(denominator)

def make_spider(df, title, color):
	# number of variable
	categories = list(df)[1:]
	N = len(categories)

	# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
	angles = [n / float(N) * 2 * pi for n in range(N)]
	angles += angles[:1]

	# Initialise the spider plot
	ax = plt.subplot(1, 1, 1, polar=True, )

	# If you want the first axis to be on top:
	ax.set_theta_offset(pi / 2)
	ax.set_theta_direction(-1)

	# Draw one axe per variable + add labels labels yet
	plt.xticks(angles[:-1], categories, color='grey', size=8)

	# Draw ylabels
	ax.set_rlabel_position(0)
	plt.yticks([5,10,15, 20, 25, 30, 35, 40], ["50","100","150", "200", "250", "300", "350", "400"], color="grey", size=10)
	#plt.yticks([250, 300, 350, 400, 450, 500], ["250", "300", "350", "400", "450", "500"], color="grey", size=10)
	plt.ylim(0, 40)

	# Ind1
	values = df.loc[0].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, color='r', linewidth=2, linestyle='solid')
	ax.fill(angles, values, color='r', alpha=0.25)

	# Ind2
	values = df.loc[1].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, color='b', linewidth=2, linestyle='solid')
	ax.fill(angles, values, color='b', alpha=0.25)

	# Ind3
	values = df.loc[2].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, color='g', linewidth=2, linestyle='solid')
	ax.fill(angles, values, color='g', alpha=0.25)
	#ax.set_title("dadad")
	# Add a title
	plt.title(title, size=11, y=1.0)
	labels = ('Boat 1', 'Boat 2', 'Boat 3')
	legend = ax.legend(labels, loc=(0.0, .95), labelspacing=0.1, fontsize='small')
	plt.close()

def search_closest_file(file_query,list_find):

	num_files_list = np.shape(list_find)[0]
	idx_found = 'b'

	for idx in range(num_files_list):
		if(abs(file_query[1]-list_find[idx][1])<=TIME_THRESHOLD):
			idx_found = idx
	
	return idx_found

def main(path_input_files) : 
	onlyfiles = [f for f in listdir(path_input_files) if isfile(join(path_input_files, f))]

	NUM_BOATS = 3
	TIME_THRESHOLD = 20

	# Step 1

	listBoat1 = []
	listBoat2 = []
	listBoat3 = []
	
	num_files_directory = np.shape(onlyfiles)[0]
	name_cut_file = 0

	for i in range(num_files_directory):
		name_cut_file = onlyfiles[i][:5]
		data_file = pd.read_csv(path_input_files+"/"+onlyfiles[i])

		if(name_cut_file == 'Boat1'):
			listBoat1.append([onlyfiles[i],data_file['TimeStamp'][0],data_file['TimeStamp'].iloc[-1]])
		elif(name_cut_file == 'Boat2'):
			listBoat2.append([onlyfiles[i],data_file['TimeStamp'][0],data_file['TimeStamp'].iloc[-1]])
		elif(name_cut_file == 'Boat3'):
			listBoat3.append([onlyfiles[i],data_file['TimeStamp'][0],data_file['TimeStamp'].iloc[-1]])
		else:
			pass

	print("Step 1: Completed!")

	# Step 2
	num_files_Boat1 = np.shape(listBoat1)[0]
	num_files_Boat2 = np.shape(listBoat2)[0]
	num_files_Boat3 = np.shape(listBoat3)[0]

	matrix_matching = []

	listBoat1_sorted = sorted(listBoat1,key=lambda x: x[1])
	#print(listBoat1_sorted)

	idx_found_list2 = 0
	idx_found_list3 = 0

	for idx in range(num_files_Boat1):
		idx_found_list2 = search_closest_file(listBoat1_sorted[idx],listBoat2)
		idx_found_list3 = search_closest_file(listBoat1_sorted[idx],listBoat3)
		if((idx_found_list2 != 'b') & (idx_found_list3 != 'b')):
			matrix_matching.append([idx,idx_found_list2,idx_found_list3])

	print("Step 2: Completed!")

	for idx_selected_matching in range(np.shape(matrix_matching)[0]):
		time_stamp_1 = [listBoat1_sorted[matrix_matching[idx_selected_matching][0]][1],listBoat2[matrix_matching[idx_selected_matching][1]][1],listBoat3[matrix_matching[idx_selected_matching][2]][1]]
		time_stamp_2 = [listBoat1_sorted[matrix_matching[idx_selected_matching][0]][2],listBoat2[matrix_matching[idx_selected_matching][1]][2],listBoat3[matrix_matching[idx_selected_matching][2]][2]]

		interval_comparison = [max(time_stamp_1),min(time_stamp_2)]

		# selected matching of each element
		data_file_boat1 = pd.read_csv(path_input_files+"/"+listBoat1_sorted[matrix_matching[idx_selected_matching][0]][0])
		data_file_boat2 = pd.read_csv(path_input_files+"/"+listBoat2[matrix_matching[idx_selected_matching][1]][0])
		data_file_boat3 = pd.read_csv(path_input_files+"/"+listBoat3[matrix_matching[idx_selected_matching][2]][0])
		
		# filter: select points on the matching interval
		new_data_file_boat1 = data_file_boat1[((data_file_boat1['TimeStamp']>=interval_comparison[0])&(data_file_boat1['TimeStamp']<=interval_comparison[1]))]
		new_data_file_boat2 = data_file_boat2[((data_file_boat2['TimeStamp']>=interval_comparison[0])&(data_file_boat2['TimeStamp']<=interval_comparison[1]))]
		new_data_file_boat3 = data_file_boat3[((data_file_boat3['TimeStamp']>=interval_comparison[0])&(data_file_boat3['TimeStamp']<=interval_comparison[1]))]

		string_idx_matching = str(idx_selected_matching)
		print(string_idx_matching.zfill(2),listBoat1_sorted[matrix_matching[idx_selected_matching][0]][0],listBoat2[matrix_matching[idx_selected_matching][1]][0],listBoat3[matrix_matching[idx_selected_matching][2]][0])
		name_boats = listBoat1_sorted[matrix_matching[idx_selected_matching][0]][0]

		# RADAR CHART
		# image_radar = draw_radar(data, array_average, array_travel,name_boats)
		data = pd.DataFrame({
			'group': ['Boat1', 'Boat2', 'Boat3'],
			'Rudder': [compute_travel(new_data_file_boat1,"Rudder")/10, compute_travel(new_data_file_boat2,"Rudder")/10, compute_travel(new_data_file_boat3,"Rudder")/10],
			'Heel':   [compute_travel(new_data_file_boat1,"Heel [Deg]")/10, compute_travel(new_data_file_boat2,"Heel [Deg]")/10, compute_travel(new_data_file_boat3,"Heel [Deg]")/10],
			'Pitch':  [compute_travel(new_data_file_boat1,"Pitch [Deg]")/10, compute_travel(new_data_file_boat2,"Pitch [Deg]")/10, compute_travel(new_data_file_boat3,"Pitch [Deg]")/10],
		})

		fig_radar = plt.figure(figsize=(7.5, 8))
		canvas = FigureCanvas(fig_radar)
		# Create a color palette:
		my_palette = plt.cm.get_cmap("Set2", len(data.index))

		fig_radar.text(0.5, 0.965, 'Boat comparison: ' + name_boats[11:17], horizontalalignment='center', color='black',
				weight='bold', size='large')

		make_spider(data, title='Travel per minute ', color=my_palette(0))

		fig_radar.canvas.draw()
		image_from_plot_radar = np.frombuffer(fig_radar.canvas.tostring_rgb(), dtype=np.uint8)
		image_from_plot_radar = image_from_plot_radar.reshape(fig_radar.canvas.get_width_height()[::-1] + (3,))

		#-------------------------------------------------------------------------------------------------------------------

		Cars = {'Brand': [1, 2, 3, 4, 5],
				'Rudder': [np.round(compute_average(new_data_file_boat1, "Rudder"),2)   , np.round(compute_average(new_data_file_boat2, "Rudder"),2)  ,np.round( compute_average(new_data_file_boat3, "Rudder"),2),5.0,-5.0],
				'Heel':   [np.round(compute_average(new_data_file_boat1, "Heel [Deg]"),2), np.round(compute_average(new_data_file_boat2, "Heel [Deg]"),2), np.round(compute_average(new_data_file_boat3, "Heel [Deg]"),2),5.0,-5.0],
				'Pitch':  [np.round(compute_average(new_data_file_boat1,"Pitch [Deg]"),2), np.round(compute_average(new_data_file_boat2,"Pitch [Deg]"),2), np.round(compute_average(new_data_file_boat3,"Pitch [Deg]"),2),5.0,-5.0]}

		iamge_pararell = parallel_plot(Cars,name_boats)

		new_image = np.hstack((iamge_pararell,image_from_plot_radar))

		cv2.imwrite('results/Bootje_' + name_boats[11:17] + '.jpg',new_image)

		data_show_travel = pd.DataFrame({
			'group': ['Boat1', 'Boat2', 'Boat3'],
			'Rudder [Trl]': [np.round(compute_travel(new_data_file_boat1,"Rudder"),2),np.round(compute_travel(new_data_file_boat2,"Rudder"),2),np.round(compute_travel(new_data_file_boat3,"Rudder"),2)],
			'Heel [Trl]':   [np.round(compute_travel(new_data_file_boat1,"Heel [Deg]"),2),np.round(compute_travel(new_data_file_boat2,"Heel [Deg]"),2),np.round(compute_travel(new_data_file_boat3,"Heel [Deg]"),2)],
			'Pitch [Trl]':  [np.round(compute_travel(new_data_file_boat1,"Pitch [Deg]"),2),np.round(compute_travel(new_data_file_boat2,"Pitch [Deg]"),2),np.round(compute_travel(new_data_file_boat3,"Pitch [Deg]"),2)],
		})

		data_show_avg = pd.DataFrame({
			'group': ['Boat1', 'Boat2', 'Boat3'],
			'Rudder [Avg]':    [np.round(compute_average(new_data_file_boat1, "Rudder"), 2),np.round(compute_average(new_data_file_boat2, "Rudder"), 2),np.round(compute_average(new_data_file_boat3, "Rudder"), 2)],
			'Heel [Avg]':      [np.round(compute_average(new_data_file_boat1, "Heel [Deg]"), 2),np.round(compute_average(new_data_file_boat2, "Heel [Deg]"), 2),np.round(compute_average(new_data_file_boat3, "Heel [Deg]"), 2)],
			'Pitch [Avg]':     [np.round(compute_average(new_data_file_boat1, "Pitch [Deg]"), 2),np.round(compute_average(new_data_file_boat2, "Pitch [Deg]"), 2),np.round(compute_average(new_data_file_boat3, "Pitch [Deg]"), 2)],
		})
		print(data_show_travel)
		print(data_show_avg)


	print("Step 3: Completed!")
	print("Hup Holland Hup!")
