#!/usr/bin/env bash


create_image_lmdb(){
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    # Please set the appropriate paths
    LMDBFILE_PATH1=$1
    IMAGEFILE_PATH1=$2
    TOOLS=$3
    DATA_ROOT=$4
    SIZE=$5
    BACKEND=$6
    # +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    # ----------------------------
    # Checks for DATA_ROOT Path
    if [ ! -d "$DATA_ROOT" ]; then
      echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
      echo "Set the DATA_ROOT variable to the path where the data instances are stored."
      exit 1
    fi

    # ------------------------------
    # Creating LMDB
     GLOG_logtostderr=1 $TOOLS/convert_imageset \
        --resize_height=$SIZE \
        --resize_width=$SIZE \
        --backend=$BACKEND \
        $DATA_ROOT \
        $IMAGEFILE_PATH1 \
        $LMDBFILE_PATH1
   # ------------------------------
    echo "Done."

}




create_vg_lmdb()
{
    # VG_shape_image lmdb creation
    echo "Imagenet attribute lmdb creation"
    LMDBFILE_PATH=/data_b/soubarna/data/visual_gnome
    IMAGEFILE_PATH=/informatik2/students/home/4banik/workspace/AttributeDetectionGCPR/data
    TOOLS=/informatik2/students/home/4banik/Documents/caffe-master2/build/tools            # Caffe tools path
    DATA_ROOT='/data_b/soubarna/data/visual_gnome/cropped_to_bbox_resized/'    # Path prefix for each entry in data.txt
    SIZE=256
    BACKEND=lmdb
    echo "creating training lmdb..."
    create_image_lmdb "$LMDBFILE_PATH/X256_VG_cropped_train_lmdb" "$IMAGEFILE_PATH/VG_X_train.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
#    echo "creating validation lmdb..."
#    create_image_lmdb "$LMDBFILE_PATH/X256_VG_cropped_val_lmdb" "$IMAGEFILE_PATH/VG_X_val.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
}

create_ycb_lmdb()
{
    echo "YCB attribute lmdb creation"
    LMDBFILE_PATH=/informatik2/students/home/4banik/workspace/AttributeDetectionGCPR/data
    IMAGEFILE_PATH=/informatik2/students/home/4banik/workspace/AttributeDetectionGCPR/data
    TOOLS=/informatik2/students/home/4banik/Documents/caffe-master2/build/tools            # Caffe tools path
    DATA_ROOT=/data_b/YCB_Video_Dataset/data/    # Path prefix for each entry in data.txt
    SIZE=256
    BACKEND=lmdb
    create_image_lmdb "$LMDBFILE_PATH/YCB_X256_train_lmdb" "$IMAGEFILE_PATH/ycb_X_train.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
    #create_image_lmdb "$LMDBFILE_PATH/YCB_X256_val_lmdb" "$IMAGEFILE_PATH/ycb_X_val.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
    
}

create_apascal_lmdb()
{
    echo "PASCAL attribute lmdb creation"
    LMDBFILE_PATH=/data_b/soubarna/data
    IMAGEFILE_PATH=/informatik2/students/home/4banik/workspace/attribute_attention/attribute_learning/data
    TOOLS=/informatik2/students/home/4banik/Documents/caffe-master2/build/tools            # Caffe tools path
    DATA_ROOT=/informatik2/students/home/4banik/Documents/datasets/VOC2008/JPEGImages/    # Path prefix for each entry in data.txt
    SIZE=256
    BACKEND=lmdb
    #create_image_lmdb "$LMDBFILE_PATH/X256_apascal_train_lmdb" "$IMAGEFILE_PATH/apascal_X_train.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
    create_image_lmdb "$LMDBFILE_PATH/X256_apascal_val_lmdb" "$IMAGEFILE_PATH/apascal_X_val.txt" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
    
}

create_ia_lmdb()
{
    echo "Imagenet attribute lmdb creation"
    LMDBFILE_OUT_PATH="$1"
    IMAGEFILE_PATH="$2"
    TOOLS="$3"            # Caffe tools path
    DATA_ROOT="$4"    # Path prefix for each entry in data.txt
    SIZE="$5"
    BACKEND="$6"
    PREFIX="$7"
    SUFFIX="$8"
    echo "creating $SUFFIX lmdb..."

    ann_file=`echo "$IMAGEFILE_PATH/$PREFIX"X_"$SUFFIX.txt"`
    out_lmdb_file=`echo "$LMDBFILE_OUT_PATH/$PREFIX"cropped_"$SUFFIX"_lmdb`

    echo $ann_file
    echo $out_lmdb_file
    create_image_lmdb "$out_lmdb_file" "$ann_file" $TOOLS $DATA_ROOT $SIZE "$BACKEND"
}


# retreive arguments
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            DS)              DS=${VALUE} ;;
            LMDBFILE_PATH)    LMDBFILE_PATH=${VALUE} ;;     
            ANNFILE_PATH)   ANNFILE_PATH=${VALUE}  ;;
            CAFFE_TOOL_PATH)  CAFFE_TOOL_PATH=${VALUE}  ;;
            DATA_ROOT)        DATA_ROOT=${VALUE} ;;

            *)   
    esac    
done

if [ "$DS" == "ia" ]
then
    PREFIX="ia_all_"
    SIZE=256
    BACKEND=lmdb
    create_ia_lmdb $LMDBFILE_PATH $ANNFILE_PATH $CAFFE_TOOL_PATH $DATA_ROOT $SIZE $BACKEND $PREFIX "train"
    if [ "$?" -eq 0 ]
    then
        echo "train lmdb created"
    else
        echo "ERROR: train lmdb not created"
    fi

    create_ia_lmdb $LMDBFILE_PATH $ANNFILE_PATH $CAFFE_TOOL_PATH $DATA_ROOT $SIZE $BACKEND $PREFIX "val"
    if [ "$?" -eq 0 ]
    then
        echo "val lmdb created"
    else
        echo "ERROR: val lmdb not created"
    fi
    
fi

