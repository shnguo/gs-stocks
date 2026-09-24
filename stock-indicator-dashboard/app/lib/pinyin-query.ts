const PINYIN_SYLLABLES = new Set(`
a ai an ang ao
ba bai ban bang bao bei ben beng bi bian biao bie bin bing bo bu
ca cai can cang cao ce cen ceng cha chai chan chang chao che chen cheng chi chong chou chu chua chuai chuan chuang chui chun chuo ci cong cou cu cuan cui cun cuo
da dai dan dang dao de dei deng di dia dian diao die ding diu dong dou du duan dui dun duo
e ei en eng er
fa fan fang fei fen feng fo fou fu
ga gai gan gang gao ge gei gen geng gong gou gu gua guai guan guang gui gun guo
ha hai han hang hao he hei hen heng hong hou hu hua huai huan huang hui hun huo
ji jia jian jiang jiao jie jin jing jiong jiu ju juan jue jun
ka kai kan kang kao ke ken keng kong kou ku kua kuai kuan kuang kui kun kuo
la lai lan lang lao le lei leng li lia lian liang liao lie lin ling liu lo long lou lu luan lue lun luo lv lve
ma mai man mang mao me mei men meng mi mian miao mie min ming miu mo mou mu
na nai nan nang nao ne nei nen neng ng ni nian niang niao nie nin ning niu nong nou nu nuan nue nuo nv nve
o ou
pa pai pan pang pao pei pen peng pi pian piao pie pin ping po pou pu
qi qia qian qiang qiao qie qin qing qiong qiu qu quan que qun
ran rang rao re ren reng ri rong rou ru rua ruan rui run ruo
sa sai san sang sao se sen seng sha shai shan shang shao she shei shen sheng shi shou shu shua shuai shuan shuang shui shun shuo si song sou su suan sui sun suo
ta tai tan tang tao te teng ti tian tiao tie ting tong tou tu tuan tui tun tuo
wa wai wan wang wei wen weng wo wu
xi xia xian xiang xiao xie xin xing xiong xiu xu xuan xue xun
ya yan yang yao ye yi yin ying yo yong you yu yuan yue yun
za zai zan zang zao ze zei zen zeng zha zhai zhan zhang zhao zhe zhei zhen zheng zhi zhong zhou zhu zhua zhuai zhuan zhuang zhui zhun zhuo zi zong zou zu zuan zui zun zuo
`.trim().split(/\s+/u));

const PINYIN_PREFIXES = new Set<string>();
for (const syllable of PINYIN_SYLLABLES) {
  for (let length = 1; length < syllable.length; length += 1) {
    PINYIN_PREFIXES.add(syllable.slice(0, length));
  }
}

type Segmentation = {
  initials: string;
  score: number;
  syllableCount: number;
};

function betterCandidate(current: Segmentation | null, next: Segmentation): Segmentation {
  if (!current) return next;
  if (next.score !== current.score) return next.score > current.score ? next : current;
  if (next.syllableCount !== current.syllableCount) {
    return next.syllableCount < current.syllableCount ? next : current;
  }
  return next.initials.length < current.initials.length ? next : current;
}

export function derivePinyinInitials(rawQuery: string): string | null {
  const query = rawQuery.trim().toLowerCase().replaceAll("ü", "v");
  if (!/^[a-z]+$/u.test(query) || query.length < 4) return null;

  const memo = new Map<number, Segmentation | null>();
  function segment(index: number): Segmentation | null {
    if (index === query.length) return { initials: "", score: 0, syllableCount: 0 };
    if (memo.has(index)) return memo.get(index) ?? null;

    let best: Segmentation | null = null;
    for (let end = index + 1; end <= query.length; end += 1) {
      const syllable = query.slice(index, end);
      if (!PINYIN_SYLLABLES.has(syllable)) continue;
      const tail = segment(end);
      if (!tail) continue;
      best = betterCandidate(best, {
        initials: syllable[0] + tail.initials,
        score: (syllable.length * syllable.length) + tail.score,
        syllableCount: tail.syllableCount + 1,
      });
    }

    const partial = query.slice(index);
    if (PINYIN_PREFIXES.has(partial)) {
      best = betterCandidate(best, {
        initials: partial[0],
        score: (partial.length * partial.length) - 2,
        syllableCount: 1,
      });
    }

    memo.set(index, best);
    return best;
  }

  const match = segment(0);
  if (!match || match.syllableCount < 2 || match.initials === query) return null;
  return match.initials;
}
