##
# Airline dataset has 13 cols + 1 for label
#   - last column is label (ArrivalDelay). Need to convert to categorical. Value > 0 means delay (1), <= 0 means no delay (0).
#   - First 13 cols are as follows.
#    (0) "Year"  => int, convert to float
#    (1) "Month" => int, convert to float
#    (2) "DayofMonth" => int, convert to float
#    (3) "DayofWeek" => int, convert to float
#    (4) "CRSDepTime" => int in hhmm format => convert to minutes and then to float
#    (5) "CRSArrTime" => int in hhmm format => convert to minutes and then to float
#    (6) "UniqueCarrier" => string. use hash map to convert to float
#    (7) "FlightNum" => int convert float
#    (8) "ActualElapsedTime" => int (in minutes) => convert to float
#    (9) "Origin" => string. use hash map to convert to float
#    (10) "Dest" => string. use same hash map as Origin to convert to float
#    (11) "Distance" => int (miles) => convert to float
#    (12) "Diverted" => int (0 or 1) => convert to float

# Note: the label (target feature) is left unprocessed - just converted to float.
# This is changed in tmp_rf_testing.cu under the RandomForest dir.

# The readme_airlines_dicts is what was used to convert categorial data to floats.
##



def find_dict_val(col_val, my_dict):
	if col_val not in my_dict:
		my_dict[col_val] = len(my_dict)
	return my_dict[col_val]


def preprocess_file(airport_dict, flight_number_dict, orig_filename, modified_filename):

	orig_file = open(orig_filename)
	output_file = open(modified_filename, 'w')

	for line in orig_file:
		line_cols = line.strip().split(",")
		new_line = ""
		for col_id in range(len(line_cols)):
			col_val = line_cols[col_id]
			if 4 <= col_id <= 5: # hhmm convert to  minutes
				hours, mins = col_val[-4:-2], col_val[-2:]
				if not mins.isdecimal():
					mins = 0
				if not hours.isdecimal():
					hours = 0
				col_val = int(hours) * 60 + int(mins)
			elif col_id == 6:	# flight number
				col_val = find_dict_val(col_val, flight_number_dict)
			elif 9 <= col_id <= 10:	# origin or dest airport
				col_val = find_dict_val(col_val, airport_dict)
			#elif col_id == 13: #label - convert to binary 1 if > 0
			#	col_val = (int(col_val) > 0)

			# Convert to float
			new_line += str(float(col_val)) + ","
		#print(new_line[:-1])
		output_file.write(new_line[:-1] + "\n")

	print("airport_dict is:")
	print(airport_dict)

	print("flight_number_dict is:")
	print(flight_number_dict)

	orig_file.close()
	output_file.close()


def invert_dict(my_dict):

	inv_dict = {}
	for key, val in my_dict.items():
		if val not in inv_dict:
			inv_dict[val] = key
		else:
			print("Error!!")
	return inv_dict

def postprocess_file(airport_dict, flight_number_dict, modified_filename, reproduced_orig_name):

	my_file = open(modified_filename)
	output_file = open(reproduced_orig_name, 'w')

	inv_flight_number_dict = invert_dict(flight_number_dict)
	inv_airport_dict = invert_dict(airport_dict)


	for line in my_file:
		line_cols = line.strip().split(",")
		new_line = ""
		for col_id in range(len(line_cols)):
			col_val = line_cols[col_id]
			if 4 <= col_id <= 5: # minutes convert to hhmm
				minutes = int(float(col_val))
				hours, mins = minutes // 60, minutes % 60
				actual_time = ""
				if hours != 0:
					actual_time += str(hours)
					if mins < 10:
						actual_time += "0"
				actual_time += str(mins)
				col_val = actual_time

			elif col_id == 6:	# flight number
				col_val = str(find_dict_val(int(float(col_val)), inv_flight_number_dict))
			elif 9 <= col_id <= 10:	# origin or dest airport
				col_val = str(find_dict_val(int(float(col_val)), inv_airport_dict))
			#elif col_id == 13: #label - convert to binary 1 if > 0
			#	col_val = (int(col_val) > 0)
			else:
				col_val = str(int(float(col_val)))

			# Convert to float
			new_line += col_val + ","
		output_file.write(new_line[:-1] + "\n")

	my_file.close()
	output_file.close()


if __name__ == "__main__":

	airport_dict = {}
	flight_number_dict = {}

	#orig_filename = "airline50"
	orig_filename = "airline_14col.data"
	modified_filename = orig_filename + "_modified"
	reproduced_orig_name = orig_filename + "_orig"

	preprocess_file(airport_dict, flight_number_dict, orig_filename, modified_filename)
	postprocess_file(airport_dict, flight_number_dict, modified_filename, reproduced_orig_name)
