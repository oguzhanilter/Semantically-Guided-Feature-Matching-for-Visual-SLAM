PATH_NAMES=("00" "02" "03" "04" "05" "06" "07" "08" "09" "10")

for i in {0..9}
    do
        ./main ${PATH_NAMES[$i]}
        
    done