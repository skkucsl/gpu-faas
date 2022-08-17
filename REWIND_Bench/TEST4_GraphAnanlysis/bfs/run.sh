# /run time on OpenWhisk Terminal

# Update actions
zip -r TEST4.zip ow_function.py
wsk action update TEST4_baseline bfs_ow.py --docker bskoon/ow_extension:python
wsk action update TEST4_rewind TEST4.zip --docker bskoon/rewind:python

# Cold start ignore
wsk action invoke TEST4_baseline --result >> /dev/null 
wsk action invoke TEST4_rewind --result >> /dev/null

# Invoke actions
echo `date` >> result4.txt
echo -en "Baseline configure\n" >> result4.txt 
wsk action invoke TEST4_baseline --result >> result4.txt

echo -en "\nREWIND configure\n" >> result4.txt
wsk action invoke TEST4_rewind --result >> result4.txt
echo -en "\n\n\n" >> result4.txt
