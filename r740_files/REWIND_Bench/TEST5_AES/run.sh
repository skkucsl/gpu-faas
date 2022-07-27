# /run time on OpenWhisk Terminal

# Update actions
zip -r TEST5.zip ow_function.py
wsk action update TEST5_baseline aes_ow.py --docker bskoon/ow_extension:python
wsk action update TEST5_rewind TEST5.zip --docker bskoon/rewind:python

# Cold start ignore
wsk action invoke TEST5_baseline --result >> /dev/null 
wsk action invoke TEST5_rewind --result >> /dev/null

# Invoke actions
echo `date` >> result5.txt
echo -en "Baseline configure\n" >> result5.txt 
wsk action invoke TEST5_baseline --result >> result5.txt

echo -en "\nREWIND configure\n" >> result5.txt
wsk action invoke TEST5_rewind --result >> result5.txt
echo -en "\n\n\n" >> result5.txt
