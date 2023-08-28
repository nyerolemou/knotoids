from enum import Enum


class KnotoidClass(Enum):
    UNCLASSIFIED = "unclassified"
    CLASS_0_1 = "0_1"
    CLASS_2_1_OR_2_1MS = "2_1|2_1ms"
    CLASS_2_1S_OR_2_1M = "2_1s|2_1m"
    CLASS_3_1 = "3_1"
    CLASS_3_1M = "3_1m"
    CLASS_3_2_OR_3_2MS = "3_2|3_2ms"
    CLASS_3_2S_OR_3_2M = "3_2s|3_2m"
    CLASS_4_1 = "4_1"
    CLASS_4_2_OR_4_2MS = "4_2|4_2ms"
    CLASS_4_2S_OR_4_2M = "4_2s|4_2m"
    CLASS_4_3_OR_4_3MS = "4_3|4_3ms"
    CLASS_4_3S_OR_4_3M = "4_3s|4_3m"
    CLASS_4_4_OR_4_4MS = "4_4|4_4ms"
    CLASS_4_4S_OR_4_4M = "4_4s|4_4m"
    CLASS_4_5_OR_4_5MS = "4_5|4_5ms"
    CLASS_4_5S_OR_4_5M = "4_5s|4_5m"
    CLASS_4_6_OR_4_6MS = "4_6|4_6ms"
    CLASS_4_6S_OR_4_6M = "4_6s|4_6m"
    CLASS_4_7_OR_4_7MS = "4_7|4_7ms"
    CLASS_4_7S_OR_4_7M = "4_7s|4_7m"
    CLASS_4_8_OR_4_8MS = "4_8|4_8ms"
    CLASS_4_8S_OR_4_8M = "4_8s|4_8m"
    CLASS_2_1x2_1_OR_2_1MSx2_1MS = "2_1*2_1|2_1ms*2_1ms"
    CLASS_2_1x2_1MS = "2_1*2_1ms"
    CLASS_2_1x2_1S_OR_2_1Mx2_1MS = "2_1*2_1s|2_1m*2_1ms"
    CLASS_2_1Sx2_1MS_OR_2_1x2_1M = "2_1s*2_1ms|2_1*2_1m"
    CLASS_2_1Sx2_1S_OR_2_1Mx2_1M = "2_1s*2_1s|2_1m*2_1m"
    CLASS_2_1Mx2_1S = "2_1m*2_1s"
    CLASS_5_1 = "5_1"
    CLASS_5_1M = "5_1m"
    CLASS_5_2 = "5_2"
    CLASS_5_2M = "5_2m"
    CLASS_5_3_OR_5_3MS = "5_3|5_3ms"
    CLASS_5_3S_OR_5_3M = "5_3s|5_3m"
    CLASS_5_4_OR_5_4MS = "5_4|5_4ms"
    CLASS_5_4S_OR_5_4M = "5_4s|5_4m"
    CLASS_5_5_OR_5_5MS = "5_5|5_5ms"
    CLASS_5_5S_OR_5_5M = "5_5s|5_5m"
    CLASS_5_6_OR_5_6MS = "5_6|5_6ms"
    CLASS_5_6S_OR_5_6M = "5_6s|5_6m"
    CLASS_5_7_OR_5_7MS = "5_7|5_7ms"
    CLASS_5_7S_OR_5_7M = "5_7s|5_7m"
    CLASS_5_8_OR_5_8MS = "5_8|5_8ms"
    CLASS_5_8S_OR_5_8M = "5_8s|5_8m"
    CLASS_5_9_OR_5_9MS = "5_9|5_9ms"
    CLASS_5_9S_OR_5_9M = "5_9s|5_9m"
    CLASS_5_10_OR_5_10MS = "5_10|5_10ms"
    CLASS_5_10S_OR_5_10M = "5_10s|5_10m"
    CLASS_5_11_OR_5_11MS = "5_11|5_11ms"
    CLASS_5_11S_OR_5_11M = "5_11s|5_11m"
    CLASS_5_12_OR_5_12MS = "5_12|5_12ms"
    CLASS_5_12S_OR_5_12M = "5_12s|5_12m"
    CLASS_5_13_OR_5_13MS = "5_13|5_13ms"
    CLASS_5_13S_OR_5_13M = "5_13s|5_13m"
    CLASS_5_14_OR_5_14MS = "5_14|5_14ms"
    CLASS_5_14S_OR_5_14M = "5_14s|5_14m"
    CLASS_5_15 = "5_15"
    CLASS_5_15M = "5_15m"
    CLASS_5_16_OR_5_16MS = "5_16|5_16ms"
    CLASS_5_16S_OR_5_16M = "5_16s|5_16m"
    CLASS_5_17 = "5_17"
    CLASS_5_17M = "5_17m"
    CLASS_5_18 = "5_18"
    CLASS_5_18M = "5_18m"
    CLASS_5_19_OR_5_19MS = "5_19|5_19ms"
    CLASS_5_19S_OR_5_19M = "5_19s|5_19m"
    CLASS_5_20_OR_5_20MS = "5_20|5_20ms"
    CLASS_5_20S_OR_5_20M = "5_20s|5_20m"
    CLASS_5_21_OR_5_21MS = "5_21|5_21ms"
    CLASS_5_21S_OR_5_21M = "5_21s|5_21m"
    CLASS_5_22_OR_5_22MS = "5_22|5_22ms"
    CLASS_5_22S_OR_5_22M = "5_22s|5_22m"
    CLASS_5_23_OR_5_23MS = "5_23|5_23ms"
    CLASS_5_23S_OR_5_23M = "5_23s|5_23m"
    CLASS_5_24_OR_5_24MS = "5_24|5_24ms"
    CLASS_5_24S_OR_5_24M = "5_24s|5_24m"
    CLASS_2_1x3_1_OR_2_1MSx3_1 = "2_1*3_1|2_1ms*3_1"
    CLASS_2_1Sx3_1_OR_2_1Mx3_1 = "2_1s*3_1|2_1m*3_1"
    CLASS_2_1x3_1M_OR_2_1MSx3_1M = "2_1*3_1m|2_1ms*3_1m"
    CLASS_2_1Sx3_1M_OR_2_1Mx3_1M = "2_1s*3_1m|2_1m*3_1m"
    CLASS_2_1x3_2_OR_2_1MSx3_2MS = "2_1*3_2|2_1ms*3_2ms"
    CLASS_2_1x3_2MS_OR_2_1MSx3_2 = "2_1*3_2ms|2_1ms*3_2"
    CLASS_2_1x3_2S_OR_2_1MSx3_2M = "2_1*3_2s|2_1ms*3_2m"
    CLASS_2_1MSx3_2S_OR_2_1x3_2M = "2_1ms*3_2s|2_1*3_2m"
    CLASS_2_1Sx3_2_OR_2_1Mx3_2MS = "2_1s*3_2|2_1m*3_2ms"
    CLASS_2_1Sx3_2MS_OR_2_1Mx3_2 = "2_1s*3_2ms|2_1m*3_2"
    CLASS_2_1Sx3_2S_OR_2_1Mx3_2M = "2_1s*3_2s|2_1m*3_2m"
    CLASS_2_1Sx3_2M_OR_2_1Mx3_2S = "2_1s*3_2m|2_1m*3_2s"
    CLASS_6_1 = "6_1"
    CLASS_6_1M = "6_1m"
    CLASS_6_2 = "6_2"
    CLASS_6_2M = "6_2m"
    CLASS_6_3 = "6_3"
    CLASS_6_4_OR_6_4MS = "6_4|6_4ms"
    CLASS_6_4S_OR_6_4M = "6_4s|6_4m"
    CLASS_6_5_OR_6_5MS = "6_5|6_5ms"
    CLASS_6_5S_OR_6_5M = "6_5s|6_5m"
    CLASS_6_6_OR_6_6MS = "6_6|6_6ms"
    CLASS_6_6S_OR_6_6M = "6_6s|6_6m"
    CLASS_6_7_OR_6_7MS = "6_7|6_7ms"
    CLASS_6_7S_OR_6_7M = "6_7s|6_7m"
    CLASS_6_8_OR_6_8MS = "6_8|6_8ms"
    CLASS_6_8S_OR_6_8M = "6_8s|6_8m"
    CLASS_6_9_OR_6_9MS = "6_9|6_9ms"
    CLASS_6_9S_OR_6_9M = "6_9s|6_9m"
    CLASS_6_10_OR_6_10MS = "6_10|6_10ms"
    CLASS_6_10S_OR_6_10M = "6_10s|6_10m"
    CLASS_6_11_OR_6_11MS = "6_11|6_11ms"
    CLASS_6_11S_OR_6_11M = "6_11s|6_11m"
    CLASS_6_12_OR_6_12MS = "6_12|6_12ms"
    CLASS_6_12S_OR_6_12M = "6_12s|6_12m"
    CLASS_6_13_OR_6_13MS = "6_13|6_13ms"
    CLASS_6_13S_OR_6_13M = "6_13s|6_13m"
    CLASS_6_14_OR_6_14MS = "6_14|6_14ms"
    CLASS_6_14S_OR_6_14M = "6_14s|6_14m"
    CLASS_6_15_OR_6_15MS = "6_15|6_15ms"
    CLASS_6_15S_OR_6_15M = "6_15s|6_15m"
    CLASS_6_16_OR_6_16MS = "6_16|6_16ms"
    CLASS_6_16S_OR_6_16M = "6_16s|6_16m"
    CLASS_6_17_OR_6_17MS = "6_17|6_17ms"
    CLASS_6_17S_OR_6_17M = "6_17s|6_17m"
    CLASS_6_18_OR_6_18MS = "6_18|6_18ms"
    CLASS_6_18S_OR_6_18M = "6_18s|6_18m"
    CLASS_6_19_OR_6_19MS = "6_19|6_19ms"
    CLASS_6_19S_OR_6_19M = "6_19s|6_19m"
    CLASS_6_20_OR_6_20MS = "6_20|6_20ms"
    CLASS_6_20S_OR_6_20M = "6_20s|6_20m"
    CLASS_6_21_OR_6_21MS = "6_21|6_21ms"
    CLASS_6_21S_OR_6_21M = "6_21s|6_21m"
    CLASS_6_22_OR_6_22MS = "6_22|6_22ms"
    CLASS_6_22S_OR_6_22M = "6_22s|6_22m"
    CLASS_6_23_OR_6_23MS = "6_23|6_23ms"
    CLASS_6_23S_OR_6_23M = "6_23s|6_23m"
    CLASS_6_24_OR_6_24MS = "6_24|6_24ms"
    CLASS_6_24S_OR_6_24M = "6_24s|6_24m"
    CLASS_6_25_OR_6_25MS = "6_25|6_25ms"
    CLASS_6_25S_OR_6_25M = "6_25s|6_25m"
    CLASS_6_26_OR_6_26MS = "6_26|6_26ms"
    CLASS_6_26S_OR_6_26M = "6_26s|6_26m"
    CLASS_6_27_OR_6_27MS = "6_27|6_27ms"
    CLASS_6_27S_OR_6_27M = "6_27s|6_27m"
    CLASS_6_28_OR_6_28MS = "6_28|6_28ms"
    CLASS_6_28S_OR_6_28M = "6_28s|6_28m"
    CLASS_6_29_OR_6_29MS = "6_29|6_29ms"
    CLASS_6_29S_OR_6_29M = "6_29s|6_29m"
    CLASS_6_30_OR_6_30MS = "6_30|6_30ms"
    CLASS_6_30S_OR_6_30M = "6_30s|6_30m"
    CLASS_6_31_OR_6_31MS = "6_31|6_31ms"
    CLASS_6_31S_OR_6_31M = "6_31s|6_31m"
    CLASS_6_32_OR_6_32MS = "6_32|6_32ms"
    CLASS_6_32S_OR_6_32M = "6_32s|6_32m"
    CLASS_6_33_OR_6_33MS = "6_33|6_33ms"
    CLASS_6_33S_OR_6_33M = "6_33s|6_33m"
    CLASS_6_34_OR_6_34MS = "6_34|6_34ms"
    CLASS_6_34S_OR_6_34M = "6_34s|6_34m"
    CLASS_6_35_OR_6_35MS = "6_35|6_35ms"
    CLASS_6_35S_OR_6_35M = "6_35s|6_35m"
    CLASS_6_36_OR_6_36MS = "6_36|6_36ms"
    CLASS_6_36S_OR_6_36M = "6_36s|6_36m"
    CLASS_6_37_OR_6_37MS = "6_37|6_37ms"
    CLASS_6_37S_OR_6_37M = "6_37s|6_37m"
    CLASS_6_38_OR_6_38MS = "6_38|6_38ms"
    CLASS_6_38S_OR_6_38M = "6_38s|6_38m"
    CLASS_6_39_OR_6_39MS = "6_39|6_39ms"
    CLASS_6_39S_OR_6_39M = "6_39s|6_39m"
    CLASS_6_40_OR_6_40MS = "6_40|6_40ms"
    CLASS_6_40S_OR_6_40M = "6_40s|6_40m"
    CLASS_6_41_OR_6_41MS = "6_41|6_41ms"
    CLASS_6_41S_OR_6_41M = "6_41s|6_41m"
    CLASS_6_42_OR_6_42MS = "6_42|6_42ms"
    CLASS_6_42S_OR_6_42M = "6_42s|6_42m"
    CLASS_6_43_OR_6_43MS = "6_43|6_43ms"
    CLASS_6_43S_OR_6_43M = "6_43s|6_43m"
    CLASS_6_44_OR_6_44MS = "6_44|6_44ms"
    CLASS_6_44S_OR_6_44M = "6_44s|6_44m"
    CLASS_6_45_OR_6_45MS = "6_45|6_45ms"
    CLASS_6_45S_OR_6_45M = "6_45s|6_45m"
    CLASS_6_46_OR_6_46MS = "6_46|6_46ms"
    CLASS_6_46S_OR_6_46M = "6_46s|6_46m"
    CLASS_6_47_OR_6_47MS = "6_47|6_47ms"
    CLASS_6_47S_OR_6_47M = "6_47s|6_47m"
    CLASS_6_48_OR_6_48MS = "6_48|6_48ms"
    CLASS_6_48S_OR_6_48M = "6_48s|6_48m"
    CLASS_6_49_OR_6_49MS = "6_49|6_49ms"
    CLASS_6_49S_OR_6_49M = "6_49s|6_49m"
    CLASS_6_50_OR_6_50MS = "6_50|6_50ms"
    CLASS_6_50S_OR_6_50M = "6_50s|6_50m"
    CLASS_6_51_OR_6_51MS = "6_51|6_51ms"
    CLASS_6_51S_OR_6_51M = "6_51s|6_51m"
    CLASS_6_52_OR_6_52MS = "6_52|6_52ms"
    CLASS_6_52S_OR_6_52M = "6_52s|6_52m"
    CLASS_6_53_OR_6_53MS = "6_53|6_53ms"
    CLASS_6_53S_OR_6_53M = "6_53s|6_53m"
    CLASS_6_54_OR_6_54S = "6_54|6_54s"
    CLASS_6_55_OR_6_55MS = "6_55|6_55ms"
    CLASS_6_55S_OR_6_55M = "6_55s|6_55m"
    CLASS_6_56_OR_6_56MS = "6_56|6_56ms"
    CLASS_6_56S_OR_6_56M = "6_56s|6_56m"
    CLASS_6_57_OR_6_57MS = "6_57|6_57ms"
    CLASS_6_57S_OR_6_57M = "6_57s|6_57m"
    CLASS_6_58_OR_6_58MS = "6_58|6_58ms"
    CLASS_6_58S_OR_6_58M = "6_58s|6_58m"
    CLASS_6_59_OR_6_59MS = "6_59|6_59ms"
    CLASS_6_59S_OR_6_59M = "6_59s|6_59m"
    CLASS_6_60_OR_6_60MS = "6_60|6_60ms"
    CLASS_6_60S_OR_6_60M = "6_60s|6_60m"
    CLASS_6_61_OR_6_61MS = "6_61|6_61ms"
    CLASS_6_61S_OR_6_61M = "6_61s|6_61m"
    CLASS_6_62 = "6_62"
    CLASS_6_62M = "6_62m"
    CLASS_6_63_OR_6_63MS = "6_63|6_63ms"
    CLASS_6_63S_OR_6_63M = "6_63s|6_63m"
    CLASS_6_64_OR_6_64MS = "6_64|6_64ms"
    CLASS_6_64S_OR_6_64M = "6_64s|6_64m"
    CLASS_6_65_OR_6_65MS = "6_65|6_65ms"
    CLASS_6_65S_OR_6_65M = "6_65s|6_65m"
    CLASS_6_66_OR_6_66MS = "6_66|6_66ms"
    CLASS_6_66S_OR_6_66M = "6_66s|6_66m"
    CLASS_6_67_OR_6_67MS = "6_67|6_67ms"
    CLASS_6_67S_OR_6_67M = "6_67s|6_67m"
    CLASS_6_68_OR_6_68MS = "6_68|6_68ms"
    CLASS_6_68S_OR_6_68M = "6_68s|6_68m"
    CLASS_6_69_OR_6_69MS = "6_69|6_69ms"
    CLASS_6_69S_OR_6_69M = "6_69s|6_69m"
    CLASS_6_70_OR_6_70MS = "6_70|6_70ms"
    CLASS_6_70S_OR_6_70M = "6_70s|6_70m"
    CLASS_6_71_OR_6_71MS = "6_71|6_71ms"
    CLASS_6_71S_OR_6_71M = "6_71s|6_71m"
    CLASS_6_72_OR_6_72MS = "6_72|6_72ms"
    CLASS_6_72S_OR_6_72M = "6_72s|6_72m"
    CLASS_6_73_OR_6_73MS = "6_73|6_73ms"
    CLASS_6_73S_OR_6_73M = "6_73s|6_73m"
    CLASS_6_74_OR_6_74MS = "6_74|6_74ms"
    CLASS_6_74S_OR_6_74M = "6_74s|6_74m"
    CLASS_6_75_OR_6_75MS = "6_75|6_75ms"
    CLASS_6_75S_OR_6_75M = "6_75s|6_75m"
    CLASS_6_76_OR_6_76MS = "6_76|6_76ms"
    CLASS_6_76S_OR_6_76M = "6_76s|6_76m"
    CLASS_6_77_OR_6_77MS = "6_77|6_77ms"
    CLASS_6_77S_OR_6_77M = "6_77s|6_77m"
    CLASS_6_78_OR_6_78MS = "6_78|6_78ms"
    CLASS_6_78S_OR_6_78M = "6_78s|6_78m"
    CLASS_6_79_OR_6_79MS = "6_79|6_79ms"
    CLASS_6_79S_OR_6_79M = "6_79s|6_79m"
    CLASS_6_80 = "6_80"
    CLASS_6_80M = "6_80m"
    CLASS_6_81_OR_6_81MS = "6_81|6_81ms"
    CLASS_6_81S_OR_6_81M = "6_81s|6_81m"
    CLASS_6_82_OR_6_82MS = "6_82|6_82ms"
    CLASS_6_82S_OR_6_82M = "6_82s|6_82m"
    CLASS_6_83_OR_6_83MS = "6_83|6_83ms"
    CLASS_6_83S_OR_6_83M = "6_83s|6_83m"
    CLASS_6_84_OR_6_84MS = "6_84|6_84ms"
    CLASS_6_84S_OR_6_84M = "6_84s|6_84m"
    CLASS_6_85 = "6_85"
    CLASS_6_85M = "6_85m"
    CLASS_6_86_OR_6_86S = "6_86|6_86s"
    CLASS_6_87 = "6_87"
    CLASS_6_87M = "6_87m"
    CLASS_6_88_OR_6_88MS = "6_88|6_88ms"
    CLASS_6_88S_OR_6_88M = "6_88s|6_88m"
    CLASS_6_89_OR_6_89MS = "6_89|6_89ms"
    CLASS_6_89S_OR_6_89M = "6_89s|6_89m"
    CLASS_6_90_OR_6_90MS = "6_90|6_90ms"
    CLASS_6_90S_OR_6_90M = "6_90s|6_90m"
    CLASS_6_91_OR_6_91MS = "6_91|6_91ms"
    CLASS_6_91S_OR_6_91M = "6_91s|6_91m"
    CLASS_6_92_OR_6_92MS = "6_92|6_92ms"
    CLASS_6_92S_OR_6_92M = "6_92s|6_92m"
    CLASS_6_93_OR_6_93MS = "6_93|6_93ms"
    CLASS_6_93S_OR_6_93M = "6_93s|6_93m"
    CLASS_6_94_OR_6_94MS = "6_94|6_94ms"
    CLASS_6_94S_OR_6_94M = "6_94s|6_94m"
    CLASS_6_95_OR_6_95MS = "6_95|6_95ms"
    CLASS_6_95S_OR_6_95M = "6_95s|6_95m"
    CLASS_6_96 = "6_96"
    CLASS_6_96M = "6_96m"
    CLASS_6_97_OR_6_97MS = "6_97|6_97ms"
    CLASS_6_97S_OR_6_97M = "6_97s|6_97m"
    CLASS_6_98_OR_6_98MS = "6_98|6_98ms"
    CLASS_6_98S_OR_6_98M = "6_98s|6_98m"
    CLASS_6_99_OR_6_99MS = "6_99|6_99ms"
    CLASS_6_99S_OR_6_99M = "6_99s|6_99m"
    CLASS_6_100_OR_6_100MS = "6_100|6_100ms"
    CLASS_6_100S_OR_6_100M = "6_100s|6_100m"
    CLASS_6_101_OR_6_101MS = "6_101|6_101ms"
    CLASS_6_101S_OR_6_101M = "6_101s|6_101m"
    CLASS_6_102_OR_6_102MS = "6_102|6_102ms"
    CLASS_6_102S_OR_6_102M = "6_102s|6_102m"
    CLASS_6_103_OR_6_103MS = "6_103|6_103ms"
    CLASS_6_103S_OR_6_103M = "6_103s|6_103m"
    CLASS_6_104_OR_6_104MS = "6_104|6_104ms"
    CLASS_6_104S_OR_6_104M = "6_104s|6_104m"
    CLASS_6_105_OR_6_105MS = "6_105|6_105ms"
    CLASS_6_105S_OR_6_105M = "6_105s|6_105m"
    CLASS_6_106_OR_6_106MS = "6_106|6_106ms"
    CLASS_6_106S_OR_6_106M = "6_106s|6_106m"
    CLASS_6_107_OR_6_107MS = "6_107|6_107ms"
    CLASS_6_107S_OR_6_107M = "6_107s|6_107m"
    CLASS_6_108_OR_6_108MS = "6_108|6_108ms"
    CLASS_6_108S_OR_6_108M = "6_108s|6_108m"
    CLASS_6_109_OR_6_109MS = "6_109|6_109ms"
    CLASS_6_109S_OR_6_109M = "6_109s|6_109m"
    CLASS_6_110_OR_6_110MS = "6_110|6_110ms"
    CLASS_6_110S_OR_6_110M = "6_110s|6_110m"
    CLASS_6_111_OR_6_111MS = "6_111|6_111ms"
    CLASS_6_111S_OR_6_111M = "6_111s|6_111m"
    CLASS_6_112_OR_6_112MS = "6_112|6_112ms"
    CLASS_6_112S_OR_6_112M = "6_112s|6_112m"
    CLASS_6_113_OR_6_113MS = "6_113|6_113ms"
    CLASS_6_113S_OR_6_113M = "6_113s|6_113m"
    CLASS_6_114_OR_6_114MS = "6_114|6_114ms"
    CLASS_6_114S_OR_6_114M = "6_114s|6_114m"
    CLASS_6_115_OR_6_115MS = "6_115|6_115ms"
    CLASS_6_115S_OR_6_115M = "6_115s|6_115m"
    CLASS_6_116_OR_6_116MS = "6_116|6_116ms"
    CLASS_6_116S_OR_6_116M = "6_116s|6_116m"
    CLASS_6_117_OR_6_117MS = "6_117|6_117ms"
    CLASS_6_117S_OR_6_117M = "6_117s|6_117m"
    CLASS_6_118_OR_6_118MS = "6_118|6_118ms"
    CLASS_6_118S_OR_6_118M = "6_118s|6_118m"
    CLASS_6_119_OR_6_119MS = "6_119|6_119ms"
    CLASS_6_119S_OR_6_119M = "6_119s|6_119m"
    CLASS_6_120_OR_6_120S = "6_120|6_120s"
    CLASS_6_121_OR_6_121MS = "6_121|6_121ms"
    CLASS_6_121S_OR_6_121M = "6_121s|6_121m"
    CLASS_2_1x4_1_OR_2_1MSx4_1 = "2_1*4_1|2_1ms*4_1"
    CLASS_2_1Sx4_1_OR_2_1Mx4_1 = "2_1s*4_1|2_1m*4_1"
    CLASS_2_1x4_2_OR_2_1MSx4_2MS = "2_1*4_2|2_1ms*4_2ms"
    CLASS_2_1x4_2MS_OR_2_1MSx4_2 = "2_1*4_2ms|2_1ms*4_2"
    CLASS_2_1x4_2S_OR_2_1MSx4_2M = "2_1*4_2s|2_1ms*4_2m"
    CLASS_2_1MSx4_2S_OR_2_1x4_2M = "2_1ms*4_2s|2_1*4_2m"
    CLASS_2_1Sx4_2_OR_2_1Mx4_2MS = "2_1s*4_2|2_1m*4_2ms"
    CLASS_2_1Sx4_2MS_OR_2_1Mx4_2 = "2_1s*4_2ms|2_1m*4_2"
    CLASS_2_1Sx4_2S_OR_2_1Mx4_2M = "2_1s*4_2s|2_1m*4_2m"
    CLASS_2_1Sx4_2M_OR_2_1Mx4_2S = "2_1s*4_2m|2_1m*4_2s"
    CLASS_2_1x4_3_OR_2_1MSx4_3MS = "2_1*4_3|2_1ms*4_3ms"
    CLASS_2_1x4_3MS_OR_2_1MSx4_3 = "2_1*4_3ms|2_1ms*4_3"
    CLASS_2_1x4_3S_OR_2_1MSx4_3M = "2_1*4_3s|2_1ms*4_3m"
    CLASS_2_1MSx4_3S_OR_2_1x4_3M = "2_1ms*4_3s|2_1*4_3m"
    CLASS_2_1Sx4_3_OR_2_1Mx4_3MS = "2_1s*4_3|2_1m*4_3ms"
    CLASS_2_1Sx4_3MS_OR_2_1Mx4_3 = "2_1s*4_3ms|2_1m*4_3"
    CLASS_2_1Sx4_3S_OR_2_1Mx4_3M = "2_1s*4_3s|2_1m*4_3m"
    CLASS_2_1Sx4_3M_OR_2_1Mx4_3S = "2_1s*4_3m|2_1m*4_3s"
    CLASS_2_1x4_4_OR_2_1MSx4_4MS = "2_1*4_4|2_1ms*4_4ms"
    CLASS_2_1x4_4MS_OR_2_1MSx4_4 = "2_1*4_4ms|2_1ms*4_4"
    CLASS_2_1x4_4S_OR_2_1MSx4_4M = "2_1*4_4s|2_1ms*4_4m"
    CLASS_2_1MSx4_4S_OR_2_1x4_4M = "2_1ms*4_4s|2_1*4_4m"
    CLASS_2_1Sx4_4_OR_2_1Mx4_4MS = "2_1s*4_4|2_1m*4_4ms"
    CLASS_2_1Sx4_4MS_OR_2_1Mx4_4 = "2_1s*4_4ms|2_1m*4_4"
    CLASS_2_1Sx4_4S_OR_2_1Mx4_4M = "2_1s*4_4s|2_1m*4_4m"
    CLASS_2_1Sx4_4M_OR_2_1Mx4_4S = "2_1s*4_4m|2_1m*4_4s"
    CLASS_2_1x4_5_OR_2_1MSx4_5MS = "2_1*4_5|2_1ms*4_5ms"
    CLASS_2_1x4_5MS_OR_2_1MSx4_5 = "2_1*4_5ms|2_1ms*4_5"
    CLASS_2_1x4_5S_OR_2_1MSx4_5M = "2_1*4_5s|2_1ms*4_5m"
    CLASS_2_1MSx4_5S_OR_2_1x4_5M = "2_1ms*4_5s|2_1*4_5m"
    CLASS_2_1Sx4_5_OR_2_1Mx4_5MS = "2_1s*4_5|2_1m*4_5ms"
    CLASS_2_1Sx4_5MS_OR_2_1Mx4_5 = "2_1s*4_5ms|2_1m*4_5"
    CLASS_2_1Sx4_5S_OR_2_1Mx4_5M = "2_1s*4_5s|2_1m*4_5m"
    CLASS_2_1Sx4_5M_OR_2_1Mx4_5S = "2_1s*4_5m|2_1m*4_5s"
    CLASS_2_1x4_6_OR_2_1MSx4_6MS = "2_1*4_6|2_1ms*4_6ms"
    CLASS_2_1x4_6MS_OR_2_1MSx4_6 = "2_1*4_6ms|2_1ms*4_6"
    CLASS_2_1x4_6S_OR_2_1MSx4_6M = "2_1*4_6s|2_1ms*4_6m"
    CLASS_2_1MSx4_6S_OR_2_1x4_6M = "2_1ms*4_6s|2_1*4_6m"
    CLASS_2_1Sx4_6_OR_2_1Mx4_6MS = "2_1s*4_6|2_1m*4_6ms"
    CLASS_2_1Sx4_6MS_OR_2_1Mx4_6 = "2_1s*4_6ms|2_1m*4_6"
    CLASS_2_1Sx4_6S_OR_2_1Mx4_6M = "2_1s*4_6s|2_1m*4_6m"
    CLASS_2_1Sx4_6M_OR_2_1Mx4_6S = "2_1s*4_6m|2_1m*4_6s"
    CLASS_2_1x4_7_OR_2_1MSx4_7MS = "2_1*4_7|2_1ms*4_7ms"
    CLASS_2_1x4_7MS_OR_2_1MSx4_7 = "2_1*4_7ms|2_1ms*4_7"
    CLASS_2_1x4_7S_OR_2_1MSx4_7M = "2_1*4_7s|2_1ms*4_7m"
    CLASS_2_1MSx4_7S_OR_2_1x4_7M = "2_1ms*4_7s|2_1*4_7m"
    CLASS_2_1Sx4_7_OR_2_1Mx4_7MS = "2_1s*4_7|2_1m*4_7ms"
    CLASS_2_1Sx4_7MS_OR_2_1Mx4_7 = "2_1s*4_7ms|2_1m*4_7"
    CLASS_2_1Sx4_7S_OR_2_1Mx4_7M = "2_1s*4_7s|2_1m*4_7m"
    CLASS_2_1Sx4_7M_OR_2_1Mx4_7S = "2_1s*4_7m|2_1m*4_7s"
    CLASS_2_1x4_8_OR_2_1MSx4_8MS = "2_1*4_8|2_1ms*4_8ms"
    CLASS_2_1x4_8MS_OR_2_1MSx4_8 = "2_1*4_8ms|2_1ms*4_8"
    CLASS_2_1x4_8S_OR_2_1MSx4_8M = "2_1*4_8s|2_1ms*4_8m"
    CLASS_2_1MSx4_8S_OR_2_1x4_8M = "2_1ms*4_8s|2_1*4_8m"
    CLASS_2_1Sx4_8_OR_2_1Mx4_8MS = "2_1s*4_8|2_1m*4_8ms"
    CLASS_2_1Sx4_8MS_OR_2_1Mx4_8 = "2_1s*4_8ms|2_1m*4_8"
    CLASS_2_1Sx4_8S_OR_2_1Mx4_8M = "2_1s*4_8s|2_1m*4_8m"
    CLASS_2_1Sx4_8M_OR_2_1Mx4_8S = "2_1s*4_8m|2_1m*4_8s"
    CLASS_3_1x3_1 = "3_1*3_1"
    CLASS_3_1x3_1M = "3_1*3_1m"
    CLASS_3_1x3_2_OR_3_1x3_2MS = "3_1*3_2|3_1*3_2ms"
    CLASS_3_1x3_2S_OR_3_1x3_2M = "3_1*3_2s|3_1*3_2m"
    CLASS_3_1Mx3_1M = "3_1m*3_1m"
    CLASS_3_1Mx3_2_OR_3_1Mx3_2MS = "3_1m*3_2|3_1m*3_2ms"
    CLASS_3_1Mx3_2S_OR_3_1Mx3_2M = "3_1m*3_2s|3_1m*3_2m"
    CLASS_3_2x3_2_OR_3_2MSx3_2MS = "3_2*3_2|3_2ms*3_2ms"
    CLASS_3_2x3_2MS = "3_2*3_2ms"
    CLASS_3_2x3_2S_OR_3_2Mx3_2MS = "3_2*3_2s|3_2m*3_2ms"
    CLASS_3_2Sx3_2MS_OR_3_2x3_2M = "3_2s*3_2ms|3_2*3_2m"
    CLASS_3_2Sx3_2S_OR_3_2Mx3_2M = "3_2s*3_2s|3_2m*3_2m"
    CLASS_3_2Mx3_2S = "3_2m*3_2s"
    CLASS_2_1x2_1x2_1_OR_2_1MSx2_1MSx2_1MS = "2_1*2_1*2_1|2_1ms*2_1ms*2_1ms"
    CLASS_2_1x2_1x2_1MS_OR_2_1x2_1MSx2_1MS = "2_1*2_1*2_1ms|2_1*2_1ms*2_1ms"
    CLASS_2_1x2_1x2_1S_OR_2_1Mx2_1MSx2_1MS = "2_1*2_1*2_1s|2_1m*2_1ms*2_1ms"
    CLASS_2_1x2_1Sx2_1MS_OR_2_1x2_1Mx2_1MS = "2_1*2_1s*2_1ms|2_1*2_1m*2_1ms"
    CLASS_2_1x2_1Sx2_1S_OR_2_1Mx2_1Mx2_1MS = "2_1*2_1s*2_1s|2_1m*2_1m*2_1ms"
    CLASS_2_1Sx2_1MSx2_1MS_OR_2_1x2_1x2_1M = "2_1s*2_1ms*2_1ms|2_1*2_1*2_1m"
    CLASS_2_1Sx2_1Sx2_1MS_OR_2_1x2_1Mx2_1M = "2_1s*2_1s*2_1ms|2_1*2_1m*2_1m"
    CLASS_2_1Sx2_1Sx2_1S_OR_2_1Mx2_1Mx2_1M = "2_1s*2_1s*2_1s|2_1m*2_1m*2_1m"
    CLASS_2_1x2_1Mx2_1S_OR_2_1Mx2_1Sx2_1MS = "2_1*2_1m*2_1s|2_1m*2_1s*2_1ms"
    CLASS_2_1Mx2_1Sx2_1S_OR_2_1Mx2_1Mx2_1S = "2_1m*2_1s*2_1s|2_1m*2_1m*2_1s"
