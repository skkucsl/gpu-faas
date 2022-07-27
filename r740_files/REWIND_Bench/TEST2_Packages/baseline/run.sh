# /run time on OpenWhisk Terminal

# Update actions
wsk action update TEST2_base1 pkg1.py --docker bskoon/ow_extension:python
wsk action update TEST2_base2 pkg2.py --docker bskoon/ow_extension:python
wsk action update TEST2_base3 pkg3.py --docker bskoon/ow_extension:python
wsk action update TEST2_base4 pkg4.py --docker bskoon/ow_extension:python

# Cold start ignore
wsk action invoke TEST2_base1 --result >> /dev/null 
wsk action invoke TEST2_base2 --result >> /dev/null
wsk action invoke TEST2_base3 --result >> /dev/null 
wsk action invoke TEST2_base4 --result >> /dev/null

# Invoke actions
echo `date` >> result2.txt
echo -en "More packages (mypy + django + sphinx + numpy)\n" >> result2.txt 
wsk action invoke TEST2_base1 --result >> result2.txt
wsk action invoke TEST2_base2 --result >> result2.txt
wsk action invoke TEST2_base3 --result >> result2.txt
wsk action invoke TEST2_base4 --result >> result2.txt
echo -en "\n\n\n" >> result2.txt
