"""
Module applying Wavelet algorithm to given signals.

March 6, 2023
"""

import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def apply_wavelet(signal, wave='db2', level=1, cutoff_low=0.7, cutoff_high=3.0, fs=30):
    """
    Given a signal, apply wavelet transform to it and return a
    resulting signal.
    """

    # apply the wavelet transform, repeatedly according the the number of levels given
    filtered_signal = _wavelet_denoise(signal, wave, level)

    # interpolate the filtered signal to match the length of the original signal
    x_old = np.linspace(0, 1, len(filtered_signal))
    x_new = np.linspace(0, 1, len(signal))
    filtered_signal = np.interp(x_new, x_old, filtered_signal)

    # # filter the interpolated signal to the desired frequency range
    # b, a = _butter_bandpass(cutoff_low, cutoff_high, fs)
    # filtered_signal = _filter_signal(filtered_signal, b, a)

    return filtered_signal


def _wavelet_denoise(signal, wavelet, level):
    
    # track signal at the end of each level
    vs = []
    sig = signal.copy()
    for _ in range(level):
        sig, cD = pywt.dwt(sig, wavelet)
        vs.append(sig)

    return vs[-1]


def _filter_signal(signal, b, a):
    """
    Filter a signal using a Butterworth filter with the given
    coefficients b and a.
    """
    filtered_signal = signal.copy()
    if len(b) == len(a) == 1:
        # if both b and a are of length 1, the filter is just a scalar multiplier
        filtered_signal = b * signal
    else:
        # apply the filter
        filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def _butter_bandpass(lowcut, highcut, fs, order=2):
    """
    Create a Butterworth bandpass filter with the given parameters.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# def apply_wavelet(signal, wave = 'db2', level = 1):
#     """
#     Given a signal, apply wavelet transform to it and return a
#     resulting signal.
#     """

#     # apply the wavelet transform, repeatedly according the the number of levels given
#     return _wavelet_denoise(signal, wave, level)

# def _wavelet_denoise(signal, wavelet, level):
    
#     # track signal at the end of each level
#     vs = []
#     sig = signal.copy()
#     for _ in range(level):
#         sig, cD = pywt.dwt(sig, wavelet)
#         vs.append(sig)

#     return vs[-1]


# sig = np.array([-0.0163918502061823, 0.05509433111492995, 0.061048769965317363, -0.031082260714613022, -0.20352091164811936, -0.3942391558717867, -0.5250443371029839, -0.538304987356054, -0.4232432872048087, -0.21829068001382715, 0.009715430495227995, 0.19596997909411895, 0.3015289050438799, 0.3212088293088825, 0.276250720560304, 0.19925204863689283, 0.1198251355736496, 0.055913112042022206, 0.011570665887997766, -0.01994044582675286, -0.0494121866265923, -0.08389562844459936, -0.12131975265134853, -0.15015248462013525, -0.1540641952741209, -0.11869890467717281, -0.03668046194440544, 0.09100613385436862, 0.2556304041535335, 0.4401319602439407, 0.6161345665477744, 0.7416208743940561, 0.7657592601404763, 0.645515215449281, 0.36985495217476194, -0.0219971775799358, -0.44212435008057693, -0.7789353350414228, -0.9361479341035682, -0.8687843620715585, -0.6004312103030065, -0.21421918713523153, 0.17841276592825064, 0.4770990167744603, 0.622974602498808, 0.6088630822526562, 0.468835891221817, 0.2562007241141105, 0.02219319507498152, -0.19600532235440726, -0.37650616450599883, -0.5075694564147918, -0.5817735300084399, -0.5930866900012226, -0.5377618587157695, -0.41802745379535416, -0.24608045214896057, -0.04538512387181645, 0.15239264969175528, 0.3141432203960218, 0.41348425493301527, 0.4375683502449762, 0.3898761671097527, 0.2874328705415933, 0.15328953815326576, 0.007811664625498982, -0.13647419670514377, -0.2732041518268642, -0.3943802285556831, -0.481761762766201, -0.5048951190677357, -0.4305287649652618, -0.24117944682626458, 0.046649494910807277, 0.3764725543160672, 0.6648386931863292, 0.8300401015958636, 0.8240147461845905, 0.6520955873010074, 0.37064121435461816, 0.06370746094499921, -0.189469802920713, -0.3408020342936675, -0.3831720479806168, -0.3429855104032333, -0.2625862471977803, -0.18218093528781043, -0.12821248381978934, -0.11014717614991248, -0.12354169683111593, -0.15564172664126347, -0.19057696003136976, -0.21318583979100392, -0.211904995820696, -0.18107672693637167, -0.12210792005629983, -0.04261190658320363, 0.046396299718959366, 0.1339694511196713, 0.2118891119096306, 0.27576250089918586, 0.3244884287478026, 0.3585952587572354, 0.3785737978573856, 0.38430247806085316, 0.3756779561405779, 0.3536336178796016, 0.32064911641975957, 0.2802074234650444, 0.23496766471147726, 0.18407143112833343, 0.12130731741807627, 0.036758746668442366, -0.07677343203181058, -0.21495927948955174, -0.3564995131236049, -0.46663563964502613, -0.5104973558951951, -0.47045215954270747, -0.3576612202692886, -0.21046093003127064, -0.0797122340123626, -0.008492403517866065, -0.015945522149357964, -0.0922375176770105, -0.20569072714444164, -0.3174993567345985, -0.3967055587921782, -0.42921359733461556, -0.4181858435188874, -0.3771001288470872, -0.31957751240555254, -0.2511211428851007, -0.1669245134769236, -0.05704389543976324, 0.08369323973906817, 0.246763586178342, 0.4078567921713142, 0.5340637218550989, 0.5975699029402692, 0.5890360957743692, 0.5235235356641581, 0.43558069203833943, 0.3655982276459139, 0.3435095104919944, 0.376576458990065, 0.4455831698690149, 0.510094816275372, 0.5209014217515217, 0.4368029030449227, 0.24161422645923525, -0.044707739278541436, -0.366799925171293, -0.6485184888249169, -0.8177137235223206, -0.8326545856855166, -0.6992878607188444, -0.4703726039914363, -0.22499398658886713, -0.03727544601830296, 0.05069624896655073, 0.03968156201358966, -0.034415448112912694, -0.11992268375091687, -0.17055226268902093, -0.16078518106769857, -0.09204806169902073, 0.010549260351911463, 0.10922333399275953, 0.16890876730207124, 0.17167133862951459, 0.12387032901613104, 0.052343838387125696, -0.008238336703385363, -0.02978027064911, -0.00016254425042225296, 0.07460985123189345, 0.17388701444748747, 0.26887048285721094, 0.3299265934763063, 0.33320790300393344, 0.2662577317120677, 0.13294381804665853, -0.04327810891618125, -0.22104663442565164, -0.3509435404276373, -0.3924758168453487, -0.33093732782278973, -0.18438948290757148, 0.003932640397881097, 0.18327762245567153, 0.31317609894382614, 0.3735205084131772, 0.36522869601040625, 0.30468761501610087, 0.21618283840631172, 0.1246013441630659, 0.04880274183220029, -0.0037193085953061075, -0.03879112856523653, -0.07199574883423543, -0.1200903044376391, -0.19139837175543709, -0.28034636559674975, -0.3688923259530778, -0.4337091938425587, -0.45558681758853803, -0.4271261446931834, -0.35546317636141955, -0.2585910558895741, -0.1570526968353656, -0.06560045015427016, 0.010605195104071818, 0.07384384949038542, 0.12816516467351974, 0.17438838039431195, 0.20884118304426297, 0.2261444400247879, 0.22377169516945483, 0.2051843248463387, 0.1795922033599614, 0.15875700284764893, 0.15262748236846982, 0.16520465062204245, 0.19147380411054898, 0.2167885980615939, 0.22051000500039675, 0.18413810922187923, 0.10092669120790865, -0.018379393551244738, -0.1477369321413975, -0.2551652451533539, -0.3142086918212962, -0.3123362626478598, -0.2540812090077541, -0.15905554371687058, -0.05564520374556181, 0.028153924508511285, 0.07396565863979004, 0.07881646642821805, 0.05507887114956389, 0.024570842399722848, 0.00904407200455315, 0.02087684317756479, 0.05834945324009261, 0.10808468851387998, 0.15304744406687865, 0.18095702269317537, 0.18815304799552213, 0.17756567848508514, 0.1536022286395971, 0.11860139994444688, 0.07363637280884985, 0.022090685939450738, -0.028519795902405443, -0.06928586263163294, -0.09519000656872278, -0.10823674972719519, -0.1159036355064833, -0.1266110413194017, -0.14570966393557683, -0.1737943978006476, -0.20661652823522425, -0.23533764412428687, -0.2473791697618659, -0.22918464684369616, -0.17103891713048253, -0.07190424600566603, 0.05836357356635352, 0.2004603495308288, 0.3298220764479548, 0.42195185804346763, 0.4576771500335725, 0.42754786407267126, 0.33449233166398984, 0.1936833256597097, 0.02917777932183893, -0.13169380306856998, -0.26414290019035347, -0.35017021014247907, -0.38075380104150236, -0.35655627774512477, -0.28776859285993045, -0.1929723750609484, -0.09627233560256614, -0.022187882424888023, 0.011018813441761088, -0.0024583851206816942, -0.05294846731962455, -0.11779830650961032, -0.16980664251606986, -0.1883715400601968, -0.16708901818052338, -0.11298356872135462, -0.038796367089084294, 0.04467911856373141, 0.1302813165483706, 0.2108318946400043, 0.2755681915319077, 0.31056370248263365, 0.3026120151523131, 0.24421054575880813, 0.13837953966902827, 0.0023635869192826665, -0.13233576209399123, -0.22653108406513006, -0.24636476205233007, -0.17793578976142657, -0.036142941729603806, 0.13792834044051372, 0.2912599450010186, 0.37875974772493626, 0.3796074374238179, 0.30289842641560233, 0.1810573106915745, 0.055558635560781944, -0.03778185022804975, -0.07880479605142643, -0.0672629240169213, -0.022023675543549504, 0.02513513938449763, 0.04133762033973886, 0.006584207788260549, -0.07715016108872656, -0.1870093345872058, -0.2892346597050435, -0.3514403822406524, -0.35119676398617017, -0.2795699143264541, -0.14206915924423433, 0.04045058230700534, 0.23162189299507652, 0.3842208058134069, 0.4527642420772088, 0.41076378716136897, 0.26326018871987733, 0.04651016879190917, -0.18519666317628397, -0.37803315015230576, -0.4935542380007775, -0.5154381620121868, -0.4487020082565347, -0.3142603458689333, -0.14097946862423716, 0.04304214733442091, 0.21714205052682856, 0.3697055052536643, 0.4942025753676129, 0.5825723376293725, 0.6205578670725085, 0.5887440819756535, 0.4704957902903375, 0.2644177084813281, -0.00492094872727003, -0.28602895602201533, -0.5138309048451529, -0.6332467028402398, -0.6209919667487624, -0.49342092070517835, -0.2971686918108145, -0.09006641205617183, 0.07661993813659485, 0.1684029864508103, 0.17086410443182576, 0.08951277767292948, -0.051793089217433044, -0.21447532967383798, -0.3525619795723972, -0.42423067293919814, -0.4052452544429517, -0.29877498297160976, -0.13541795072660973, 0.0385820409707511, 0.17988226582207623, 0.2638437908556707, 0.28976376324777003, 0.27408848185400286, 0.2368031752719939, 0.18926597871644774, 0.13006126620745379, 0.050508876974648065, -0.053749009638841216, -0.1716157537193298, -0.2723873998654487, -0.3125969012512326, -0.25319240228288253, -0.08057911491913222, 0.17899602012470178, 0.46121154838755973, 0.6838282363660609, 0.7771693538158823, 0.7105736000890233, 0.5033608403536954, 0.21800511418949414, -0.05950083391915567, -0.2455104606563043, -0.28494174586160376, -0.17121437491285807, 0.04771285559273755, 0.28241161505352375, 0.4325655724042457, 0.42520503832153855, 0.2452752915653592, -0.05621820199150507, -0.38278368001368396, -0.6333994347134992, -0.7431923966487445, -0.7033494329291279, -0.5532223453262766, -0.3538262104580758, -0.15978720974662286, -0.0034488568580976486, 0.1051816118762772, 0.16875172439719452, 0.19293870848189582, 0.18289622237021042, 0.1451387625552081, 0.09055998803817428, 0.03483722612603801, -0.005278752451937763, -0.018931242687005967, -0.007906857282976798, 0.012019402208094221, 0.018234490440961974, -0.006192920512750777, -0.06290826941339855, -0.13811744890161975, -0.21092619596560247, -0.263320112387332, -0.2854372891719409, -0.27465724431341065, -0.23149550717247944, -0.15666664791534568, -0.051990680330461336, 0.07548367985530968, 0.20959957776806418, 0.3254001734945975, 0.39524374028897646, 0.39860604661313, 0.3307351503522703, 0.20561812938344964, 0.05192120523238948, -0.09570279843485996, -0.20583285519180053, -0.2568562884489224, -0.2400441239858248, -0.16006128446306023, -0.03403188610764751, 0.11074580471722473, 0.24120952222494602, 0.3254642488487528, 0.34107895313397135, 0.28210368863099977, 0.16199904683529498, 0.01093229374655269, -0.13217117746651763, -0.23052050403540958, -0.2598051482650824, -0.21468264433525364, -0.10947733653660452, 0.02702579481033318, 0.1609307095616057, 0.263284071124729, 0.31691449073018463, 0.3177767724118675, 0.27158142390922957, 0.18894670692656518, 0.08248301095489458, -0.03303384824545402, -0.13931353533477575, -0.21547430111835683, -0.24336259108940136, -0.21590013219990017, -0.14378657093352448, -0.05535894507146055, 0.012434712496823153, 0.0287753271347293, -0.017649625711621764, -0.1133068067493852, -0.22461962361037202, -0.3099177462078573, -0.3334052348216899, -0.2768744088364253, -0.14661326737252073, 0.02576387830250535, 0.1891114752405768, 0.2853844882016119, 0.2684265408244343, 0.1250111418975615, -0.11091583550176654, -0.36171684327087766, -0.5314133794900971, -0.544213708679338, -0.3783470438816689, -0.07880025361388777, 0.2570073617167169, 0.5142912131258959, 0.6025510916984634, 0.48831088261327876, 0.20750018537629783, -0.1465555367679305, -0.45513020317808484, -0.614760021687406, -0.5719810154370392, -0.3394423887452155, 0.012017036506991707, 0.3808122314715751, 0.6695932205350077, 0.814530245548994, 0.8000929426404817, 0.6569535770559733, 0.44605675883638257, 0.23549943185601957, 0.07838418480848489, -0.0014377856528932087, -0.012304850730709582, 0.01328843768698662, 0.03443069239286309, 0.019725172428228105, -0.03934684373794506, -0.12532737615834197, -0.20369986960058026, -0.24016385371943766, -0.2184128284588035, -0.14971755454067623, -0.06878504928386586, -0.01690435870100304, -0.02075192622702743, -0.07880563255851403, -0.16350944837889575, -0.2375557164236762, -0.2735031881772342, -0.2645902574865344, -0.22236251473727164, -0.16665280767992827, -0.11662950737855399, -0.08676052985257382, -0.08545787939021615, -0.11311571052704772, -0.15982115207656739, -0.20629820550391675, -0.23062478952482857, -0.21833808836695578, -0.16951900228679728, -0.09808523110352058, -0.02471454101160575, 0.03099857437486625, 0.05550995295958566, 0.044350210482109964, 0.005135750290660046, -0.04076584088194324, -0.06177621801844185, -0.026720630064425377, 0.08069209934762361, 0.2506184983497602, 0.44590070455656067, 0.6134594339861004, 0.70439420767931, 0.6935099547457995, 0.588088999350855, 0.42132558628921246, 0.2350229845664858, 0.062285958576054894, -0.08047384182283346, -0.19038285173508696, -0.26956511159558905, -0.31906297884676155, -0.3388625280416996, -0.331687333674275, -0.3055728644486052, -0.2720077104823676, -0.2400257147179033, -0.20933770194461515, -0.16670152073851757, -0.08934236182851646, 0.043400194225605315, 0.23409419852174662, 0.4556561542382135, 0.653442533751635, 0.7643720277155666, 0.7463841644260965, 0.6028913240353783, 0.38634877742940377, 0.1754658327395212, 0.03657984977615347, -0.009682202593498113, 0.0021739748290306865, 0.0018118976907299383, -0.08027162401992127, -0.2787898231933382, -0.5788900410170058, -0.9245856264377151, -1.2421528785066782, -1.4652167002741392, -1.5511888646818084, -1.4857751802200085, -1.2785637095402203, -0.9551895879788022, -0.5503493972602105, -0.10321745473503777])
# # plt.plot(sig[sig > 0])
# s1 = apply_wavelet(sig, level = 1)
# s2 = apply_wavelet(sig, level = 2)
# s3 = apply_wavelet(sig, level = 3)
# plt.plot(s1[s1 > 0])
# # plt.plot(s2[s2 > 0])
# plt.plot(s3[s3 > 0])
# plt.show()
