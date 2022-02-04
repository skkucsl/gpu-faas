import subprocess

def main(args):
	temp = subprocess.check_output("pmap -x `cut -d ' ' -f 4 /proc/self/stat`", shell=True)
	return {'pmap_result': str(temp)}
	#print("{\'pmap_result\': \'",temp,"\'}")
