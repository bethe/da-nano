"""
Your task is to check the "productionStartYear" of the DBPedia autos datafile for valid values.
The following things should be done:
- check if the field "productionStartYear" contains a year - done
- check if the year is in range 1886-2014 - done
- convert the value of the field to be just a year (not full datetime) - done
- the rest of the fields and values should stay the same - done
- if the value of the field is a valid year in the range as described above,
  write that line to the output_good file - done
- if the value of the field is not a valid year as described above,
  write that line to the output_bad file - done
- discard rows (neither write to good nor bad) if the URI is not from
  dbpedia.org - done
- you should use the provided way of reading and writing data (DictReader and DictWriter)
  They will take care of dealing with the header.

You can write helper functions for checking the data and writing the files, but we will call only the
'process_file' with 3 arguments (inputfile, output_good, output_bad).
"""
import csv
import pprint
import re

INPUT_FILE = 'autos.csv'
OUTPUT_GOOD = 'autos-valid.csv'
OUTPUT_BAD = 'FIXME-autos.csv'

def process_file(input_file, output_good, output_bad):

    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        # - discard rows (neither write to good nor bad) if the URI is not from dbpedia.org
        valid_rows = [row for row in reader if row['URI'].startswith("http://dbpedia.org") and row not in header]
        # - check if the field "productionStartYear" contains a year - done
        valid_years = [row for row in valid_rows if re.search('[0-9]{4}.*',row["productionStartYear"]) is not None]
        # - check if the year is in range 1886-2014
        valid_years = [row for row in valid_years if int(row['productionStartYear'][0:4]) in range (1886,2014)]
        # - transform year format to only four years
        for year in valid_years:
            year['productionStartYear'] = year['productionStartYear'][0:4]
        # - Store invalid entries to write out later in OUTPUT_BAD
        bad_years = [line for line in valid_rows if line not in valid_years]



    # This is just an example on how you can use csv.DictWriter
    # Remember that you have to output 2 files
    with open(output_good, "w") as g:
        writer = csv.DictWriter(g, delimiter=",", fieldnames= header)
        writer.writeheader()
        for row in valid_years:
            writer.writerow(row)

    with open(output_bad, "w") as g:
        writer = csv.DictWriter(g, delimiter=",", fieldnames= header)
        writer.writeheader()
        for row in bad_years:
            writer.writerow(row)

def test():

    process_file(INPUT_FILE, OUTPUT_GOOD, OUTPUT_BAD)


if __name__ == "__main__":
    test()
