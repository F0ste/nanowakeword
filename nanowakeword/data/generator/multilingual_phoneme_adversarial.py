# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

import re
import random
from typing import List, Dict, Optional, Tuple

# Chinese support
try:
    from pypinyin import pinyin, Style
    _PINYIN_AVAILABLE = True
except ImportError:
    _PINYIN_AVAILABLE = False

# Cantonese support
try:
    import jyutping
    _JYUTPING_AVAILABLE = True
except ImportError:
    _JYUTPING_AVAILABLE = False


def _require_pypinyin():
    if not _PINYIN_AVAILABLE:
        raise RuntimeError(
            "pypinyin is required for Chinese adversarial generation. "
            "Install it with: pip install pypinyin"
        )


def _require_jyutping():
    if not _JYUTPING_AVAILABLE:
        raise RuntimeError(
            "jyutping is required for Cantonese adversarial generation. "
            "Install it with: pip install jyutping"
        )


class ChinesePhonemeAdversarialGenerator:
    """
    Generate Chinese adversarial phrases using pinyin-based phonetic substitution.

    Converts the wake word to pinyin, substitutes initials/finals/tones with
    phonetically similar alternatives, then maps back to real Chinese characters
    that the TTS engine can pronounce.

    Parameters
    ----------
    min_distance : float, optional
        Minimum phonetic distance (0.0-1.0) between original and generated text.
        Higher values produce more distinct variants. Default 0.30.
    """

    # Initial consonant similarity groups (by place/manner of articulation)
    INITIAL_GROUPS: Dict[str, List[str]] = {
        # Bilabial
        'b': ['b', 'p', 'm'],
        'p': ['p', 'b', 'm'],
        'm': ['m', 'b', 'p', 'f'],
        # Labiodental
        'f': ['f', 'h'],
        # Alveolar stops/nasals/lateral
        'd': ['d', 't', 'n', 'l'],
        't': ['t', 'd', 'n', 'l'],
        'n': ['n', 'l', 'd', 't'],
        'l': ['l', 'n', 'd', 't', 'r'],
        # Velar
        'g': ['g', 'k', 'h'],
        'k': ['k', 'g', 'h'],
        'h': ['h', 'f', 'g', 'k'],
        # Alveolo-palatal
        'j': ['j', 'q', 'x'],
        'q': ['q', 'j', 'x'],
        'x': ['x', 'j', 'q'],
        # Retroflex
        'zh': ['zh', 'ch', 'sh', 'z', 'c', 's'],
        'ch': ['ch', 'zh', 'sh', 'c', 'z', 's'],
        'sh': ['sh', 'zh', 'ch', 's', 'z', 'c', 'r'],
        'r': ['r', 'sh', 'l', 'n'],
        # Dental sibilants
        'z': ['z', 'c', 's', 'zh', 'ch', 'sh'],
        'c': ['c', 'z', 's', 'ch', 'zh', 'sh'],
        's': ['s', 'z', 'c', 'sh', 'zh', 'ch'],
        # Zero initial (vowel-initial)
        '': ['', 'y', 'w'],
    }

    # Final (rhyme) similarity groups
    FINAL_GROUPS: Dict[str, List[str]] = {
        # Single vowels
        'a': ['a', 'ia', 'ua', 'e', 'ai', 'ao', 'an', 'ang'],
        'o': ['o', 'uo', 'ou', 'ong', 'e'],
        'e': ['e', 'ie', 'ue', 'er', 'ei', 'en', 'eng', 'a', 'o'],
        'i': ['i', 'in', 'ing', 'ei', 'ui'],
        'u': ['u', 'ou', 'ong', 'iu', 'ui', 'un'],
        'v': ['v', 'u', 'i', 've'],

        # Compound finals
        'ai': ['ai', 'ei', 'an', 'a', 'ao'],
        'ei': ['ei', 'ai', 'en', 'e', 'ui'],
        'ao': ['ao', 'ou', 'ang', 'a', 'iao'],
        'ou': ['ou', 'ao', 'ong', 'o', 'iu'],

        # Nasal finals
        'an': ['an', 'ang', 'en', 'ian', 'uan', 'ai'],
        'en': ['en', 'eng', 'ei', 'in', 'un'],
        'ang': ['ang', 'an', 'eng', 'ong', 'iang', 'uang'],
        'eng': ['eng', 'en', 'ang', 'ong', 'ing'],
        'ong': ['ong', 'eng', 'ang', 'iong', 'ou'],
        'in': ['in', 'ing', 'en', 'ian'],
        'ing': ['ing', 'in', 'eng', 'iang'],
        'ian': ['ian', 'iang', 'an', 'uan', 'in'],
        'iang': ['iang', 'ian', 'ang', 'uang', 'ing'],
        'uan': ['uan', 'uang', 'an', 'ian', 'un'],
        'uang': ['uang', 'uan', 'ang', 'iang', 'ong'],
        'un': ['un', 'ong', 'en', 'iong', 'uan'],
        'iong': ['iong', 'ong', 'un', 'iang'],
        'ia': ['ia', 'ie', 'a', 'ian'],
        'ie': ['ie', 'ia', 'e', 've', 'ian'],
        'ua': ['ua', 'uo', 'a', 'uai', 'uan'],
        'uo': ['uo', 'ua', 'o', 'uai', 'ong'],
        'uai': ['uai', 'uai', 'ai', 'ua', 'uan'],
        'ui': ['ui', 'ei', 'iu', 'un', 'u'],
        'iu': ['iu', 'ou', 'ui', 'u', 'iong'],
        've': ['ve', 'ie', 'v', 'e'],
        'er': ['er', 'e', 'en', 'a'],
    }

    # Tone number to Pinyin tone mark mapping
    _TONE_MAP = {1: '\u0304', 2: '\u0301', 3: '\u030c', 4: '\u0300', 5: ''}

    # Most common 3500 Chinese characters with pinyin for reverse lookup
    _CHAR_PINYIN_CACHE: Optional[Dict[str, List[Tuple[str, int]]]] = None

    def __init__(self, min_distance: float = 0.12):
        _require_pypinyin()
        self.min_distance = max(0.0, min(1.0, min_distance))
        if self._CHAR_PINYIN_CACHE is None:
            self._build_char_pinyin_cache()

    @classmethod
    def _build_char_pinyin_cache(cls):
        """Build a pinyin -> list of (character, tone) reverse index."""
        cls._CHAR_PINYIN_CACHE = {}

        common_chars = cls._get_common_chars()
        for ch in common_chars:
            try:
                results = pinyin(ch, style=Style.TONE3, heteronym=True)
                for pron_list in results:
                    for pron in pron_list:
                        s = re.sub(r'\d+', '', pron)
                        t_match = re.search(r'(\d)', pron)
                        tone = int(t_match.group(1)) if t_match else 5
                        cls._CHAR_PINYIN_CACHE.setdefault(s, []).append((ch, tone))
            except Exception:
                continue

    @staticmethod
    def _get_common_chars() -> str:
        """Return a string of most common Chinese characters (3500)."""
        return (
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话术米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片却始专状育厂京识适属圆包火住调满县局照参红细引听该铁价严"
            "龙致首底官富纪限效依优倒坐粉敌略客袁冷胜绝析块剂丝协重诉念陈仍罗盐友洋错苦夜刑移频逐靠混念母短皮终聚汽村云哪既距卫停烈央察烧迅境若印洲刻括激孔搞甚室待核校散侵吧甲游久菜味旧模湖货损预阻毫普稳乙妈植息扩银语挥酒守拿序纸医缺雨吗针刘啊急唱误训愿审附获难茶鲜粮斤孩脱硫肥善龙演父渐血欢械掌歌沙著刚攻谓盾讨晚粒乱燃矛乎杀药宁鲁贵钟煤读班伯香介迫句丰培握兰担弦蛋沉假穿执答乐谁顺烟缩征脸喜松脚困异免背星福买染井概慢怕磁倍祖皇促静补评翻肉践尼衣宽扬棉希伤操垂秋宜氢套督振架亮末宪庆编牛套附触映雷销诗座居抓裂胞呼娘景威绿晶厚盟衡鸡孙延危胶屋乡临陆顾掉扦灯岁措束耐剧玉赵跳哥季课凯井彪印潮鲁裂预输述材肩陆荡荣忧死唱宁晨鸡倍扫审等等"
            "吗吧嗯哦呢啊了么着过了的吧呢吗啊与和而及或但且如以则因为所以如果虽然但是然而并且而且"
            "零一二三四五六七八九十百千万亿兆"
            "年月日时分秒今天明昨前午晚早晨晚午间点鐘"
            "东西南北上下左右前后里外中内旁"
            "大小多少长短高矮胖瘦新旧好坏真假虚实"
            "红黄蓝绿白黑紫橘粉灰棕金銀色"
            "天空气日月星云风雨雪雷电冰霜露雾"
            "山水河海湖江溪泉渊池波浪潮涛"
            "花草树木叶根茎芽果种子土泥石沙金銀铜铁锡"
            "人男女老幼婴儿孩童青少年壮中老翁婆"
            "头脸眼耳鼻口牙舌喉咙脖子肩膀胸背腰腹"
            "手臂掌指腿脚膝趾皮肤血肉骨筋脈"
            "心思感情爱恨喜怒哀乐悲恐惊忧烦惱"
            "说看听吃喝闻摸走跑跳坐站躺睡醒"
            "家房屋门窗户墙壁楼梯厨厕浴卧厅堂"
            "桌椅凳床櫃架箱包袋布衣裤鞋帽"
            "饭菜单汤肉蛋奶酒茶水果蔬菜米面"
            "学校教室图书馆操场实验室老师学生课本书笔纸"
            "工农商学兵医律师艺术音乐体育舞蹈"
            "城乡村镇市省国界街道巷路桥隧道"
            "车船飞机火车公交自行车摩托卡车"
            "手机电脑电视机电冰箱洗衣空调微波炉"
            "钱金银财宝贵重物品商店市场超市銀行"
            "春夏秋冬春夏秋冬季节气候温度热冷暖"
            "动植生物猫狗鸡鸭鱼鸟虫蛇马牛猪羊兔"
            "乐快慢硬软干湿冷热明暗新旧富贫"
            "钟点刻秒钟表时间光陰古今未现在"
            "语词句段落篇章文诗词歌曲赋辞"
            "教习学问知识智慧才能技巧方法"
            "政军法制度规矩法律条例命令决议"
            "科数理化天地生医农工程技术"
            "礼义廉耻慈孝敬信仁爱和平"
        )

    def _parse_pinyin(self, syllable: str) -> Tuple[str, str, int]:
        """Parse a pinyin syllable into (initial, final, tone)."""
        consonant_map = [
            'zh', 'ch', 'sh',
            'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
            'g', 'k', 'h', 'j', 'q', 'x',
            'r', 'z', 'c', 's',
            'y', 'w',
        ]

        s = syllable.rstrip('012345')
        tone_match = re.search(r'(\d)', syllable)
        tone = int(tone_match.group(1)) if tone_match else 5

        initial = ''
        final = s
        for cons in consonant_map:
            if s.startswith(cons):
                initial = cons
                final = s[len(cons):]
                break

        if final and final[0] in ('y', 'w') and initial == '':
            initial = final[0]
            final = final[1:]

        if initial == 'y':
            initial = ''
            if final.startswith('u'):
                final = 'v' + final[1:]
            elif final.startswith('i'):
                final = 'i' + final[1:]
        elif initial == 'w':
            initial = ''
            if final.startswith('u'):
                final = 'u' + final[1:]

        return initial, final, tone

    def get_pinyin(self, text: str) -> List[Tuple[str, str, str, int]]:
        """Get pinyin components for each character: (original_char, initial, final, tone)."""
        results = []
        pinyin_list = pinyin(text, style=Style.TONE3)
        for i, (char, py_list) in enumerate(zip(text, pinyin_list)):
            py = py_list[0] if py_list else char
            init, fin, tone = self._parse_pinyin(py)
            results.append((char, init, fin, tone))
        return results

    def _initial_distance(self, i1: str, i2: str) -> float:
        if i1 == i2:
            return 0.0
        if i1 in self.INITIAL_GROUPS and i2 in self.INITIAL_GROUPS[i1]:
            return 0.3
        return 0.7

    def _final_distance(self, f1: str, f2: str) -> float:
        if f1 == f2:
            return 0.0
        if f1 in self.FINAL_GROUPS and f2 in self.FINAL_GROUPS[f1]:
            return 0.3
        return 0.7

    def _tone_distance(self, t1: int, t2: int) -> float:
        if t1 == t2:
            return 0.0
        diff = abs(t1 - t2)
        return min(diff * 0.2, 0.6)

    def calculate_distance(self, pinyin1: List[Tuple[str, str, int]], pinyin2: List[Tuple[str, str, int]]) -> float:
        """Calculate phonetic distance between two pinyin sequences (0.0-1.0)."""
        if not pinyin1 or not pinyin2:
            return 1.0

        max_len = max(len(pinyin1), len(pinyin2))
        total_diff = 0.0
        count = 0

        for idx in range(max_len):
            if idx >= len(pinyin1) or idx >= len(pinyin2):
                total_diff += 0.8
                count += 1
                continue

            item1 = pinyin1[idx]
            item2 = pinyin2[idx]
            if len(item1) == 4:
                i1, f1, t1 = item1[1], item1[2], item1[3]
            else:
                i1, f1, t1 = item1[0], item1[1], item1[2]
            if len(item2) == 4:
                i2, f2, t2 = item2[1], item2[2], item2[3]
            else:
                i2, f2, t2 = item2[0], item2[1], item2[2]

            diff = (
                self._initial_distance(i1, i2) * 0.35 +
                self._final_distance(f1, f2) * 0.40 +
                self._tone_distance(t1, t2) * 0.25
            )
            total_diff += diff
            count += 1

        return total_diff / count if count > 0 else 0.0

    def _get_similar_characters(
        self, char: str, init: str, fin: str, tone: int, num_variants: int = 30
    ) -> List[str]:
        """Find characters with similar pronunciation to the given pinyin."""
        candidates = set()
        char_set = set()

        init_group = self.INITIAL_GROUPS.get(init, [init])
        final_group = self.FINAL_GROUPS.get(fin, [fin])

        tone_variants = [tone]
        for d in [1, 2, 3]:
            if 1 <= tone + d <= 5:
                tone_variants.append(tone + d)
            if 1 <= tone - d <= 5:
                tone_variants.append(tone - d)

        cache = self._CHAR_PINYIN_CACHE
        if cache is None:
            return []

        for alt_init in init_group:
            for alt_fin in final_group:
                for alt_tone in tone_variants:
                    syllable = alt_init + alt_fin
                    if syllable in cache:
                        for c, t in cache[syllable]:
                            if t == alt_tone and c != char:
                                candidates.add(c)

        return list(candidates)[:num_variants]

    def generate(self, input_text: str, n: int = 10) -> List[str]:
        """
        Generate N phonetically similar but distinct Chinese phrases.

        Args:
            input_text: Original wake-word text (Chinese characters).
            n: Number of adversarial variants to generate.

        Returns:
            List of Chinese text strings.
        """
        _require_pypinyin()
        if not input_text or n <= 0:
            return []

        original_pinyin = self.get_pinyin(input_text)
        if not original_pinyin:
            return []

        variants = []
        seen = {input_text}
        attempts = 0
        max_attempts = n * 80

        num_chars = len(original_pinyin)

        while len(variants) < n and attempts < max_attempts:
            attempts += 1

            min_changes = max(1, int(num_chars * 0.30))
            max_changes = max(min_changes + 1, int(num_chars * 0.70))
            change_count = random.randint(min_changes, max_changes)

            changeable_indices = list(range(num_chars))
            random.shuffle(changeable_indices)
            indices_to_change = sorted(changeable_indices[:change_count])

            new_chars = list(input_text)
            new_pinyin = list(original_pinyin)

            for idx in indices_to_change:
                char, init, fin, tone = original_pinyin[idx]
                similar = self._get_similar_characters(char, init, fin, tone)
                if similar:
                    chosen = random.choice(similar)
                    new_chars[idx] = chosen
                    try:
                        py_result = pinyin(chosen, style=Style.TONE3)
                        if py_result and py_result[0]:
                            new_parsed = self._parse_pinyin(py_result[0][0])
                            new_pinyin[idx] = (chosen,) + new_parsed
                    except Exception:
                        new_pinyin[idx] = original_pinyin[idx]
                else:
                    change_type = random.choice(['init', 'fin', 'tone'])
                    if change_type == 'init':
                        init_group = self.INITIAL_GROUPS.get(init, [init])
                        alt_init = random.choice([x for x in init_group if x != init])
                        alt_fin = fin
                        alt_tone = tone
                    elif change_type == 'fin':
                        fin_group = self.FINAL_GROUPS.get(fin, [fin])
                        alt_init = init
                        alt_fin = random.choice([x for x in fin_group if x != fin])
                        alt_tone = tone
                    else:
                        alt_init = init
                        alt_fin = fin
                        alt_tone = random.choice([t for t in [1, 2, 3, 4, 5] if t != tone])

                    syllable = alt_init + alt_fin
                    cache = self._CHAR_PINYIN_CACHE
                    if cache and syllable in cache:
                        matches = [(c, t) for c, t in cache[syllable] if t == alt_tone and c != char]
                        if matches:
                            chosen = random.choice(matches)[0]
                            new_chars[idx] = chosen
                            new_pinyin[idx] = (chosen, alt_init, alt_fin, alt_tone)
                        else:
                            new_chars[idx] = input_text[idx]
                    else:
                        new_chars[idx] = input_text[idx]

            result = ''.join(new_chars)
            if result == input_text or result in seen:
                continue

            new_pinyin_filtered = []
            for item in new_pinyin:
                if len(item) == 4:
                    new_pinyin_filtered.append(item[1:])
                else:
                    new_pinyin_filtered.append(item)

            orig_pinyin_tuples = [x[1:] for x in original_pinyin]

            distance = self.calculate_distance(orig_pinyin_tuples, new_pinyin_filtered)
            if distance < self.min_distance:
                continue

            variants.append(result)
            seen.add(result)

        return variants[:n]


class KoreanPhonemeAdversarialGenerator:
    """
    Generate Korean adversarial phrases using jamo decomposition and substitution.

    Decomposes Korean hangul into jamo, substitutes phonetically similar jamo,
    and recombines into valid Korean syllables that TTS can pronounce.

    Parameters
    ----------
    min_distance : float, optional
        Minimum phonetic distance (0.0-1.0) between original and generated text.
        Default 0.30.
    """

    # Jamo tables for decomposition
    _CHOSEONG = [
        'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
        'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]
    _JUNGSEONG = [
        'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
        'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
    ]
    _JONGSEONG = [
        '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ',
        'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ',
        'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
    ]

    # Jamo phonetic substitution maps
    CHO_SUBSTITUTIONS: Dict[str, List[str]] = {
        'ㄱ': ['ㄲ', 'ㅋ', 'ㅇ', 'ㅎ'],
        'ㄲ': ['ㄱ', 'ㅋ'],
        'ㄴ': ['ㄷ', 'ㄸ', 'ㄹ', 'ㅌ'],
        'ㄷ': ['ㄸ', 'ㅌ', 'ㄴ', 'ㄹ', 'ㅈ', 'ㅊ'],
        'ㄸ': ['ㄷ', 'ㅌ'],
        'ㄹ': ['ㄴ', 'ㄷ'],
        'ㅁ': ['ㅂ', 'ㅃ', 'ㅍ'],
        'ㅂ': ['ㅃ', 'ㅍ', 'ㅁ'],
        'ㅃ': ['ㅂ', 'ㅍ'],
        'ㅅ': ['ㅆ', 'ㅈ', 'ㅊ', 'ㅎ'],
        'ㅆ': ['ㅅ', 'ㅈ', 'ㅊ'],
        'ㅇ': ['ㅎ', 'ㄱ'],
        'ㅈ': ['ㅉ', 'ㅊ', 'ㅅ', 'ㄷ', 'ㅌ'],
        'ㅉ': ['ㅈ', 'ㅊ'],
        'ㅊ': ['ㅈ', 'ㅉ', 'ㅅ', 'ㅌ'],
        'ㅋ': ['ㄱ', 'ㄲ', 'ㅎ'],
        'ㅌ': ['ㄷ', 'ㄸ', 'ㅊ'],
        'ㅍ': ['ㅂ', 'ㅃ', 'ㅁ'],
        'ㅎ': ['ㅇ', 'ㄱ', 'ㅋ'],
    }

    JUNG_SUBSTITUTIONS: Dict[str, List[str]] = {
        'ㅏ': ['ㅑ', 'ㅓ', 'ㅐ', 'ㅘ'],
        'ㅐ': ['ㅒ', 'ㅔ', 'ㅏ', 'ㅙ'],
        'ㅑ': ['ㅏ', 'ㅕ', 'ㅒ'],
        'ㅒ': ['ㅐ', 'ㅖ', 'ㅑ'],
        'ㅓ': ['ㅕ', 'ㅏ', 'ㅔ', 'ㅝ'],
        'ㅔ': ['ㅖ', 'ㅐ', 'ㅓ', 'ㅞ'],
        'ㅕ': ['ㅓ', 'ㅑ', 'ㅖ'],
        'ㅖ': ['ㅔ', 'ㅒ', 'ㅕ'],
        'ㅗ': ['ㅛ', 'ㅜ', 'ㅚ', 'ㅘ'],
        'ㅘ': ['ㅙ', 'ㅏ', 'ㅗ'],
        'ㅙ': ['ㅚ', 'ㅐ', 'ㅘ'],
        'ㅚ': ['ㅙ', 'ㅗ', 'ㅟ'],
        'ㅛ': ['ㅗ', 'ㅠ', 'ㅚ'],
        'ㅜ': ['ㅠ', 'ㅗ', 'ㅟ', 'ㅝ'],
        'ㅝ': ['ㅞ', 'ㅓ', 'ㅜ'],
        'ㅞ': ['ㅝ', 'ㅔ', 'ㅟ'],
        'ㅟ': ['ㅚ', 'ㅜ', 'ㅞ'],
        'ㅠ': ['ㅜ', 'ㅛ', 'ㅟ'],
        'ㅡ': ['ㅣ', 'ㅜ'],
        'ㅢ': ['ㅡ', 'ㅣ', 'ㅔ'],
        'ㅣ': ['ㅡ', 'ㅢ'],
    }

    def __init__(self, min_distance: float = 0.06):
        self.min_distance = max(0.0, min(1.0, min_distance))
        self._SBase = 0xAC00
        self._LBase = 0x1100
        self._VBase = 0x1161
        self._TBase = 0x11A7
        self._LCount = len(self._CHOSEONG)
        self._VCount = len(self._JUNGSEONG)
        self._TCount = len(self._JONGSEONG)
        self._NCount = self._VCount * self._TCount
        self._SCount = self._LCount * self._NCount

    def decompose(self, text: str) -> List[Tuple[str, str, str, str]]:
        """
        Decompose Korean text into jamo per syllable.
        Returns list of (original_syllable, choseong, jungseong, jongseong).
        Non-hangul characters are passed through unchanged.
        """
        results = []
        for ch in text:
            code = ord(ch)
            if 0xAC00 <= code <= 0xD7A3:
                sindex = code - self._SBase
                l_index = sindex // self._NCount
                v_index = (sindex % self._NCount) // self._TCount
                t_index = sindex % self._TCount
                cho = self._CHOSEONG[l_index]
                jung = self._JUNGSEONG[v_index]
                jong = self._JONGSEONG[t_index]
                results.append((ch, cho, jung, jong))
            else:
                results.append((ch, ch, '', ''))
        return results

    def compose(self, cho: str, jung: str, jong: str) -> str:
        """Compose a single Korean syllable from jamo."""
        try:
            l_index = self._CHOSEONG.index(cho)
            v_index = self._JUNGSEONG.index(jung)
            t_index = self._JONGSEONG.index(jong) if jong else 0
            code = self._SBase + (l_index * self._NCount) + (v_index * self._TCount) + t_index
            return chr(code)
        except (ValueError, IndexError):
            return cho + jung + jong

    def calculate_distance(
        self, jamo1: List[Tuple[str, str, str, str]], jamo2: List[Tuple[str, str, str, str]]
    ) -> float:
        """Calculate phonetic distance between two jamo sequences."""
        if not jamo1 or not jamo2:
            return 1.0

        max_len = max(len(jamo1), len(jamo2))
        total_diff = 0.0
        count = 0

        for idx in range(max_len):
            if idx >= len(jamo1) or idx >= len(jamo2):
                total_diff += 0.8
                count += 1
                continue

            _, c1, j1, t1 = jamo1[idx]
            _, c2, j2, t2 = jamo2[idx]

            cho_diff = 0.0 if c1 == c2 else (0.3 if c2 in self.CHO_SUBSTITUTIONS.get(c1, []) else 0.7)
            jung_diff = 0.0 if j1 == j2 else (0.3 if j2 in self.JUNG_SUBSTITUTIONS.get(j1, []) else 0.7)
            jong_diff = 0.0 if t1 == t2 else (0.3 if t1 in self._JONGSEONG and t2 in self._JONGSEONG else 0.7)

            diff = cho_diff * 0.5 + jung_diff * 0.35 + jong_diff * 0.15
            total_diff += diff
            count += 1

        return total_diff / count if count > 0 else 0.0

    def generate(self, input_text: str, n: int = 10) -> List[str]:
        """
        Generate N phonetically similar but distinct Korean phrases.

        Args:
            input_text: Original wake-word text (Korean hangul).
            n: Number of adversarial variants to generate.

        Returns:
            List of Korean text strings.
        """
        if not input_text or n <= 0:
            return []

        original_jamo = self.decompose(input_text)
        if not original_jamo:
            return []

        variants = []
        seen = {input_text}
        attempts = 0
        max_attempts = n * 80

        num_syllables = len(original_jamo)

        while len(variants) < n and attempts < max_attempts:
            attempts += 1

            min_changes = max(1, int(num_syllables * 0.30))
            max_changes = max(min_changes + 1, int(num_syllables * 0.70))
            change_count = random.randint(min_changes, max_changes)

            changeable = [i for i, s in enumerate(original_jamo) if ord(s[0]) >= 0xAC00]
            if not changeable:
                break

            count = min(change_count, len(changeable))
            indices = sorted(random.sample(changeable, count))

            result_chars = []
            new_jamo = list(original_jamo)

            for idx, (syll, cho, jung, jong) in enumerate(original_jamo):
                if idx in indices and ord(syll) >= 0xAC00:
                    change_type = random.choice(['cho', 'jung', 'jong', 'cho_jung', 'jung_jong'])

                    new_cho = cho
                    new_jung = jung
                    new_jong = jong

                    if 'cho' in change_type and cho in self.CHO_SUBSTITUTIONS:
                        subs = [s for s in self.CHO_SUBSTITUTIONS[cho] if s != cho]
                        if subs:
                            new_cho = random.choice(subs)

                    if 'jung' in change_type and jung in self.JUNG_SUBSTITUTIONS:
                        subs = [s for s in self.JUNG_SUBSTITUTIONS[jung] if s != jung]
                        if subs:
                            new_jung = random.choice(subs)

                    if 'jong' in change_type:
                        possible = [j for j in self._JONGSEONG if j != jong]
                        if possible:
                            new_jong = random.choice(possible)

                    composed = self.compose(new_cho, new_jung, new_jong)
                    result_chars.append(composed)
                    new_jamo[idx] = (composed, new_cho, new_jung, new_jong)
                else:
                    result_chars.append(syll)

            result = ''.join(result_chars)
            if result == input_text or result in seen:
                continue

            distance = self.calculate_distance(original_jamo, new_jamo)
            if distance < self.min_distance:
                continue

            variants.append(result)
            seen.add(result)

        return variants[:n]


class CantonesePhonemeAdversarialGenerator:
    """
    Generate Cantonese adversarial phrases using jyutping-based phonetic substitution.

    Uses the same Chinese character output as Mandarin but relies on Cantonese TTS
    for pronunciation. Since both Mandarin and Cantonese wake words share the same
    characters (你好酒店), this generates character-level substitutions that are
    phonetically similar in their respective reading.

    Parameters
    ----------
    min_distance : float, optional
        Minimum phonetic distance (0.0-1.0). Default 0.30.
    """

    def __init__(self, min_distance: float = 0.12):
        self.min_distance = max(0.0, min(1.0, min_distance))
        self._mandarin_gen = ChinesePhonemeAdversarialGenerator(min_distance=min_distance)

    def generate(self, input_text: str, n: int = 10) -> List[str]:
        """
        Generate N phonetically similar Cantonese adversarial phrases.

        Since Cantonese wake words use the same Chinese characters as Mandarin,
        this delegates to the Chinese generator. The Cantonese TTS engine will
        pronounce the characters with Cantonese readings.

        Args:
            input_text: Original wake-word text (Chinese characters).
            n: Number of adversarial variants.

        Returns:
            List of Chinese text strings for Cantonese TTS.
        """
        return self._mandarin_gen.generate(input_text, n)


def create_generator(lang: str, min_distance: float = 0.12):
    """
    Factory to create the appropriate generator for a given language.

    Args:
        lang: Language code ('zh', 'ko', 'yue', 'cantonese', 'korean').
        min_distance: Minimum phonetic distance (0.0-1.0).

    Returns:
        A generator instance.
    """
    lang = lang.lower().strip()
    if lang in ('zh', 'chinese', 'mandarin', 'yue', 'cantonese'):
        return ChinesePhonemeAdversarialGenerator(min_distance=min_distance)
    elif lang in ('ko', 'korean'):
        return KoreanPhonemeAdversarialGenerator(min_distance=min_distance)
    else:
        raise ValueError(f"Unsupported language: {lang}. Supported: zh, ko, yue")
