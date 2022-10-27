for i in $(seq 1 10)
do
./a.out > /dev/null
done
dmesg | grep "PAGE_FAULT:" | tail -n 10
