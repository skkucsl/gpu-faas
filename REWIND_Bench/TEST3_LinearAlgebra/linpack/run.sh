# /run time on OpenWhisk Terminal

# Update actions
zip -r TEST3.zip ow_function.py
wsk action update TEST3_baseline linpack_ow.py --docker bskoon/ow_extension:python
wsk action update TEST3_rewind TEST3.zip --docker bskoon/rewind:python

# Cold start ignore
wsk action invoke TEST3_baseline --result >> /dev/null 
wsk action invoke TEST3_rewind --result >> /dev/null

# Invoke actions
echo `date` >> result3.txt
echo -en "Baseline configure\n" >> result3.txt 
wsk action invoke TEST3_baseline --result >> result3.txt

echo -en "\nREWIND configure\n" >> result3.txt
wsk action invoke TEST3_rewind --result >> result3.txt
echo -en "\n\n\n" >> result3.txt
