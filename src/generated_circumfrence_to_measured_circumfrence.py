import statistics

#https://www.smf.org/docs/articles/pdf/chingtechbrief.pdf ||table 1.|| Head Data
#
#
person1 = [53.7 , 18.4, 14.3]
person2 = [53.3 , 18.1, 14.3]
person3 = [54.3 , 15.0, 11.2] #how is the length shorter, and the breadth shorter result in a larger measured circumference.
person4 = [54.0 , 18.0, 15.0]
person5 = [52.0 , 18.3, 13.8]
person6 = [54.0 , 18.5, 14.0]
person7 = [52.7 , 17.4, 14.3]
person8 = [53.0 , 18.0, 14.2]
person9 = [60.3 , 25.0, 15.5]
person10 = [56.4 , 18.7, 15.1]
person11 = [58.5 , 20.0, 15.7]
person12 = [55.7 , 18.9, 15.9]
person13 = [55.3 , 18.6, 14.6]
person14 = [55.2 , 18.7, 15.3]
person15 = [60.8 , 19.7, 16.4]

#Divides the measured circumference with the square circumference using the length and breadth data.
def measured_circumference_percent_of_generated_circumfrence(measured_circumference, length, breadth):
    box_circumference = (length * 2)+(breadth * 2)
    percent = measured_circumference / box_circumference
    #print("%.2f" % box_circumference, measured_circumference, str("%.2f" % percent))
    #print (str(percent))
    return percent

#Making head objects that can be used to solve for the mean with the difference in scale with a decimal percentage.
head_1 = (measured_circumference_percent_of_generated_circumfrence(*person1))
head_2 = (measured_circumference_percent_of_generated_circumfrence(*person2))
head_3 = (measured_circumference_percent_of_generated_circumfrence(*person3))
head_4 = (measured_circumference_percent_of_generated_circumfrence(*person4))
head_5 = (measured_circumference_percent_of_generated_circumfrence(*person5))
head_6 = (measured_circumference_percent_of_generated_circumfrence(*person6))
head_7 = (measured_circumference_percent_of_generated_circumfrence(*person7))
head_8 = (measured_circumference_percent_of_generated_circumfrence(*person8))
head_9 = (measured_circumference_percent_of_generated_circumfrence(*person9))
head_10 = (measured_circumference_percent_of_generated_circumfrence(*person10))
head_11 = (measured_circumference_percent_of_generated_circumfrence(*person11))
head_12 = (measured_circumference_percent_of_generated_circumfrence(*person12))
head_13 = (measured_circumference_percent_of_generated_circumfrence(*person13))
head_14 = (measured_circumference_percent_of_generated_circumfrence(*person14))
head_15 = (measured_circumference_percent_of_generated_circumfrence(*person15))
#Using the imported statistics to solve for the mean of floats in the list of heads.
#Removed outlier results to improve accuracy |head_3,
list_of_heads = [head_1 , head_2 ,  head_4 , head_5 , head_6 , head_7 , head_8 , head_9 , head_10 , head_11 , head_12 , head_13 , head_14 , head_15]
mean_head_percentage = statistics.mean(list_of_heads)
print(mean_head_percentage)
#This transcendental constant is used to express the gamma function and can be used to find arc length of elipse gausss_constant = 0.834626841674
#print("this is the head avg percentage {}".format(mean_head_percentage))
#Using the mean difference percentage stored in mean_head_percentage it then corrects the boxed circumference returns generated_circumference.
def generated_circumference_x_mean_head_percentage(measured_circumference, length, breadth,mean_head_percentage):
    generated_circumference_no_correction = (length * 2)+(breadth * 2)
    ##using gauss_constant to see if accuracy improves || off by -31.74 mm in dataset|| reverting to trained percentage in mean_head_percentage
    generated_circumference_corrected_w_avg_head_percentage = generated_circumference_no_correction * mean_head_percentage
    #print(generated_circumference_no_correction)
    #print (generated_circumference_corrected_w_avg_head_percentage)
    return generated_circumference_corrected_w_avg_head_percentage


#Using the known measured circumference we subtract the circumference generated with length times breadth (generated_circumference).
#If the number is negative it means that the generated circumference was longer than the measured circumference.
#If the number is positive it means that the generated circumference was shorter than measured circumference.
def generated_circumference_minus_measured_diffrence(measured_circumfrence, length, breadth,mean_head_percentage):
    generated_circumference = generated_circumference_x_mean_head_percentage(measured_circumfrence, length, breadth, mean_head_percentage)
    difference =  measured_circumfrence - generated_circumference
    #print (difference)
    return difference

#Making test objects that contain the difference between the measured and the generated measurement for circumference.
test1 = generated_circumference_minus_measured_diffrence(*person1, mean_head_percentage)
test2 = generated_circumference_minus_measured_diffrence(*person2, mean_head_percentage)
test3 = generated_circumference_minus_measured_diffrence(*person3, mean_head_percentage)
test4 = generated_circumference_minus_measured_diffrence(*person4, mean_head_percentage)
test5 = generated_circumference_minus_measured_diffrence(*person5, mean_head_percentage)
test6 = generated_circumference_minus_measured_diffrence(*person6, mean_head_percentage)
test7 = generated_circumference_minus_measured_diffrence(*person7, mean_head_percentage)
test8 = generated_circumference_minus_measured_diffrence(*person8, mean_head_percentage)
test9 = generated_circumference_minus_measured_diffrence(*person9, mean_head_percentage)
test10 = generated_circumference_minus_measured_diffrence(*person10, mean_head_percentage)
test11 = generated_circumference_minus_measured_diffrence(*person11, mean_head_percentage)
test12 = generated_circumference_minus_measured_diffrence(*person12, mean_head_percentage)
test13 = generated_circumference_minus_measured_diffrence(*person13, mean_head_percentage)
test14 = generated_circumference_minus_measured_diffrence(*person14, mean_head_percentage)
test15 = generated_circumference_minus_measured_diffrence(*person15, mean_head_percentage)
list_of_tests = [test1, test2 ,  test4 , test5 , test6 , test7 , test8, test9, test10 , test11, test12 , test13 , test14 , test15]
#removed test3,

mean_diffrence_box_measured = (statistics.mean(list_of_tests))
#^^^if you remove the abs on the mean above remove the # before #print() to determine if the generated circumference is shorter or longer than the measured circumference
#print("If the number is negative it means that the generated circumference was longer than the measured circumference.")
#print("If the number is positive it means that the generated circumference was shorter than measured circumference.")
mean_diffrence_generated_cm = (mean_diffrence_box_measured*2.54)
print("This is the mean between the difference of the measured circumference and the generated circumference: {} inches || {} mm".format(mean_diffrence_box_measured , mean_diffrence_generated_cm*10))
print("This would determine that the generated circumference is off by : +/- {} inches || +/- {} mm".format(mean_diffrence_box_measured , mean_diffrence_generated_cm*10))
