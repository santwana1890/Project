  *	v?V??@2Z
#Iterator::Root::Prefetch::Generatorqr?CQ ??!???j?/T@)qr?CQ ??1???j?/T@:Preprocessing2O
Iterator::Root::Prefetch??:?Ϝ?!h_???@)??:?Ϝ?1h_???@:Preprocessing2E
Iterator::Root;s	????!J&??~ @)?????'??1-?K?!@:Preprocessing2?
SIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeatJy???!z???H?@)?Ѫ?t???1??oO@:Preprocessing2V
Iterator::Root::Prefetch::Shard?????Μ?!??.?}?@)?w?'-\??127??;[??:Preprocessing2?
]Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????r-??!şF?+???)?b)????1?`?q=??:Preprocessing2s
<Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2`??-}?!F?Œ?1??)`??-}?1F?Œ?1??:Preprocessing2d
-Iterator::Root::Prefetch::Shard::Rebatch::Map3??p?܌?!?H?ҭ@)[?? ??|?1??^g???:Preprocessing2x
AIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip&?2????!,M?h?
@)ض(?A&y?1+??\????:Preprocessing2?
_Iterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??
???s?!E7?F????)??
???s?1E7?F????:Preprocessing2?
mIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?PۆQp?!1~x?(<??)?PۆQp?11~x?(<??:Preprocessing2_
(Iterator::Root::Prefetch::Shard::RebatchƧ Ϡ??![`?8]?@)aU??N?i?1???/{%??:Preprocessing2?
MIterator::Root::Prefetch::Shard::Rebatch::Map::ParallelMapV2::Zip[0]::FlatMap???N??!S??8??@)?mO???^?1?+??;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.