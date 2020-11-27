
import csv

inputfile = csv.reader(open('housing.csv','r'))
count_nyc = 0
#  40 listings less than 400 price and 200 sqft

with open('nyc_housing.csv', mode='w') as employee_file:
    print("processing...\n")
    nyc_housing = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    nyc_housing.writerow(['id','region', 'price','type','sqfeet','beds','baths','cats_allowed','dogs_allowed','smoking_allowed','wheelchair_access','electric_vehicle_charge','comes_furnished','laundry_options','parking_options','lat','long','state'])

    # cleaning data from urls and descriptions
    for row in inputfile:
        # and (float(row[6]) <= 200 or float(row[4]) <= 400)
        if row[2] == "new york city" and (float(row[6]) <= 200 or float(row[4]) <= 400):
                print("Skipping an out of range house with a price of ", row[4], " and SQFT of ", row[6])
        # and float(row[4]) > 400 and float(row[6]) > 200
        if row[2] == "new york city" and float(row[4]) > 400 and float(row[6]) > 200:
            # print(row)
            nyc_housing.writerow([row[0], row[2], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[19], row[20], row[21]])
            count_nyc +=1

print("\nNYC Lisitings processed, number of listings:\t", count_nyc)
