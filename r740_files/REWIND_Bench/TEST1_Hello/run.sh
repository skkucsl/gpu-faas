# /run time on OpenWhisk Terminal

# Update actions
zip -r TEST1.zip ow_function.py
wsk action update TEST1_baseline hello_ow.py
wsk action update TEST1_rewind TEST1.zip --docker bskoon/rewind:python

# Cold start ignore
wsk action invoke TEST1_baseline --result >> /dev/null 
wsk action invoke TEST1_rewind --result >> /dev/null

# Invoke actions
echo `date` >> result1.txt
echo -en "Baseline configure\n" >> result1.txt 
wsk action invoke TEST1_baseline --result >> result1.txt

echo -en "\nREWIND configure\n" >> result1.txt
wsk action invoke TEST1_rewind --result >> result1.txt
echo -en "\n\n\n" >> result1.txt
