# sh setup_data_path.sh data_path dataset
data_path=$1
dataset=$2

if [[ ${dataset} == "domainnet40"  ]] ;
then
  cd domainnet40
  rm clipart
  ln -s "${data_path}/clipart" clipart
  rm infograph
  ln -s "${data_path}/infograph" infograph
  rm painting
  ln -s "${data_path}/painting" painting
  rm quickdraw
  ln -s "${data_path}/quickdraw" quickdraw
  rm real
  ln -s "${data_path}/real" real
  rm sketch
  ln -s "${data_path}/sketch" sketch
  cd ..
elif [[ ${dataset} == "office31"  ]] ;
then
  cd office31
  rm amazon
  ln -s "${data_path}/amazon" amazon
  rm webcam
  ln -s "${data_path}/webcam" webcam
  rm dslr
  ln -s "${data_path}/dslr" dslr
elif [[ ${dataset} == "office-home"  ]] ;
then
  cd office-home
  rm Art
  ln -s "${data_path}/Art" Art
  rm Clipart
  ln -s "${data_path}/Clipart" Clipart
  rm Product
  ln -s "${data_path}/Product" Product
  rm Real_World
  ln -s "${data_path}/Real_World" Real_World
elif [[ ${dataset} == "office-home-rsut"  ]] ;
then
  cd office-home-rsut
  rm Art
  ln -s "${data_path}/Art" Art
  rm Clipart
  ln -s "${data_path}/Clipart" Clipart
  rm Product
  ln -s "${data_path}/Product" Product
  rm Real_World
  ln -s "${data_path}/Real_World" Real_World
elif [[ ${dataset} == "visda"  ]] ;
then
  cd visda-2017
  rm train
  ln -s "${data_path}/train" train
  rm validation
  ln -s "${data_path}/validation" validation
fi
cd ..