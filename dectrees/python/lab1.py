import numpy as np
import matplotlib.pyplot as plt
import monkdata as md
import dtree as d
import drawtree_qt5 as dtqt
import math

#Calculate the entropy
monk1 = d.entropy(md.monk1)
monk2 = d.entropy(md.monk2)
monk3 = d.entropy(md.monk3)

print ('Entropy calculation:')
print ('The entropy of monk 1 is: ' + str(monk1))
print ('The entropy of monk 2 is: ' + str(monk2))
print ('The entropy of monk 3 is: ' + str(monk3))

#Information gain
gainM1 = np.empty([6,1], dtype = float)
gainM2 = np.empty([6,1], dtype = float)
gainM3 = np.empty([6,1], dtype = float)

for i in range(0,6):
    gainM1[i] = d.averageGain(md.monk1, md.attributes[i])
    gainM2[i] = d.averageGain(md.monk2, md.attributes[i])
    gainM3[i] = d.averageGain(md.monk3, md.attributes[i])

print ('Information gain for monk 1 from a1 to a6: ' + str(np.transpose(gainM1)))
print ('Information gain for monk 2 from a1 to a6: ' + str(np.transpose(gainM2)))
print ('Information gain for monk 3 from a1 to a6: ' + str(np.transpose(gainM3)))

#Split monk1 and check best information gain for next attributes
gainSecondM1 = np.empty([6,4], dtype = float)

for i in range(0,4):
    for j in range (0,6):
        gainSecondM1[j,i] = d.averageGain(d.select(md.monk1, md.attributes[4], (i+1)), md.attributes[j])

print ('Information gain for monk 1 in the next level of the tree for each value: ' + str(np.transpose(gainSecondM1)))

#Build a tree
tree1 = d.buildTree(md.monk1, md.attributes)
tree2 = d.buildTree(md.monk2, md.attributes)
tree3 = d.buildTree(md.monk3, md.attributes)

#dtqt.drawTree(tree1)

print ('Train and test set errors for the three Monk datasets for the full trees:')
print ('Monk1:')
print((1 - d.check(tree1,md.monk1)))
print((1 - d.check(tree1,md.monk1test)))

print ('Monk2:')
print((1 - d.check(tree2,md.monk2)))
print((1 - d.check(tree2,md.monk2test)))

print ('Monk3:')
print((1 - d.check(tree3,md.monk3)))
print((1 - d.check(tree3,md.monk3test)))

#
fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
monk1Err, monk1Var = d.reducedErrorPrune(md.monk1, md.monk1test, md.attributes, fractions, 500)
monk3Err, monk3Var = d.reducedErrorPrune(md.monk3, md.monk3test, md.attributes, fractions, 500)
monk1Dev = np.empty([6,1], dtype = float)
monk3Dev = np.empty([6,1], dtype = float)
for v in range(0,6):
    monk1Dev[v] = math.sqrt(monk1Var[v])
    monk3Dev[v] = math.sqrt(monk3Var[v])

print('Monk1 Mean Error: ' + str(monk1Err))
print('Monk1 Standard Deviation: ' + str(monk1Dev))
print('Monk3 Mean Error: ' + str(monk3Err))
print('Monk3 Standard Deviation: ' + str(monk3Dev))

plt.figure(figsize=(7, 7))

plt.subplot(2, 1, 1)
plt.title('Mean Error as a function of fraction, 500 simulations')
plt.xlabel('Fraction')
plt.ylabel('Mean Error')
line1, = plt.plot(fractions, monk1Err, 'bo-', label="Monk 1")
plt.legend(handles = [line1])
plt.tight_layout(5.0)

plt.subplot(2, 1, 2)
plt.title('Mean Error as a function of fraction, 500 simulations')
plt.xlabel('Fraction')
plt.ylabel('Mean Error')
line1, = plt.plot(fractions, monk3Err, '-ro', label="Monk 3")
plt.legend(handles = [line1])

plt.show()
