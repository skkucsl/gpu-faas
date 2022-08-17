# /run time on OpenWhisk Terminal

# Make zip files
cp pkg1.py ow_function.py
zip -r pkg1.zip ow_function.py
cp pkg2.py ow_function.py
zip -r pkg2.zip ow_function.py
cp pkg3.py ow_function.py
zip -r pkg3.zip ow_function.py
cp pkg4.py ow_function.py
zip -r pkg4.zip ow_function.py

# Update actions
wsk action update TEST2_rewind1 pkg1.zip --docker bskoon/rewind:python
wsk action update TEST2_rewind2 pkg2.zip --docker bskoon/rewind:python
wsk action update TEST2_rewind3 pkg3.zip --docker bskoon/rewind:python
wsk action update TEST2_rewind4 pkg4.zip --docker bskoon/rewind:python

# Cold start ignore
wsk action invoke TEST2_rewind1 --result >> /dev/null 
wsk action invoke TEST2_rewind2 --result >> /dev/null
wsk action invoke TEST2_rewind3 --result >> /dev/null 
wsk action invoke TEST2_rewind4 --result >> /dev/null

# Invoke actions
echo `date` >> result2.txt
echo -en "More packages (mypy + django + sphinx + numpy)\n" >> result2.txt 
wsk action invoke TEST2_rewind1 --result >> result2.txt
wsk action invoke TEST2_rewind2 --result >> result2.txt
wsk action invoke TEST2_rewind3 --result >> result2.txt
wsk action invoke TEST2_rewind4 --result >> result2.txt
echo -en "\n\n\n" >> result2.txt
