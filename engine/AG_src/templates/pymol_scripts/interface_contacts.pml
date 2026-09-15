# =============================================================================
# interface_contacts.pml
# 수용체-펩타이드 계면 상호작용 분석 렌더링
# Interface contacts analysis: H-bonds, salt bridges, hydrophobic contacts
#
# 사용법 / Usage:
#   pymol -c interface_contacts.pml
# 변수 / Variables (파이프라인에서 치환):
#   @RECEPTOR_PDB@  : 수용체 PDB 경로
#   @PEPTIDE_PDB@   : 펩타이드 PDB 경로
#   @OUTPUT_PNG@    : 출력 PNG 경로
# =============================================================================

reinitialize

# --- 구조 로드 ---
load @RECEPTOR_PDB@, receptor
load @PEPTIDE_PDB@, peptide

# --- 기본 표현 설정 ---
hide everything
show cartoon, receptor
color gray80, receptor
set cartoon_transparency, 0.5, receptor

show sticks, peptide
color hotpink, peptide
set stick_radius, 0.15

# --- 계면 잔기 선택 (4 Å 접촉 기준) ---
select iface_receptor, byres (receptor within 4 of peptide)
select iface_peptide, byres (peptide within 4 of receptor)

show sticks, iface_receptor
show sticks, iface_peptide
util.cbac iface_receptor         # carbon = cyan
util.cbam iface_peptide          # carbon = magenta

# =============================================================================
# 1. 수소결합 (Hydrogen Bonds)
#    mode=2: 공여체-수용체 쌍 사이의 수소결합
# =============================================================================
distance hbonds, iface_peptide, iface_receptor, 3.5, mode=2
color blue, hbonds
set dash_color, blue, hbonds
set dash_gap, 0.25, hbonds
set dash_width, 2.5, hbonds
set dash_radius, 0.05, hbonds

# =============================================================================
# 2. 염교 (Salt Bridges)
#    양전하(Arg, Lys) <-> 음전하(Asp, Glu) 잔기 간 거리 4 Å 이내
# =============================================================================
select pos_charged, (iface_peptide or iface_receptor) and (resn ARG+LYS) and (name NH1+NH2+NZ)
select neg_charged, (iface_peptide or iface_receptor) and (resn ASP+GLU) and (name OD1+OD2+OE1+OE2)
distance salt_bridges, pos_charged, neg_charged, 4.0, mode=0
color red, salt_bridges
set dash_color, red, salt_bridges
set dash_gap, 0.3, salt_bridges
set dash_width, 2.0, salt_bridges

# =============================================================================
# 3. 소수성 접촉 (Hydrophobic Contacts)
#    소수성 잔기 탄소 원자 간 거리 4.5 Å 이내
# =============================================================================
select hydrophobic, (iface_peptide or iface_receptor) and (resn ALA+VAL+ILE+LEU+MET+PHE+TRP+PRO+TYR) and name C*
distance hydrophobic_contacts, hydrophobic and iface_peptide, hydrophobic and iface_receptor, 4.5, mode=0
color orange, hydrophobic_contacts
set dash_color, orange, hydrophobic_contacts
set dash_gap, 0.4, hydrophobic_contacts
set dash_width, 1.5, hydrophobic_contacts

# --- 거리 레이블 표시 (수소결합 거리 값) ---
set label_size, 12
set label_color, blue
# hbonds 객체에만 레이블 표시
label hbonds, "%.1f" % distance

# --- 계면 잔기 이름 레이블 ---
set label_size, 11
set label_color, black
label (iface_receptor and name CA), "%s%s" % (oneletter, resi)
label (iface_peptide and name CA), "%s%s" % (oneletter, resi)

# --- 범례 (Legend) - 주석으로 기록 ---
# Blue dashes   = Hydrogen bonds (H-bond, <=3.5 Å)
# Red dashes    = Salt bridges (<=4.0 Å)
# Orange dashes = Hydrophobic contacts (<=4.5 Å)

# --- 카메라 설정 ---
zoom iface_receptor, 6
orient iface_receptor

# --- 렌더링 ---
set ray_shadows, 1
set ambient, 0.45
set antialias, 2
bg_color white

ray 2400, 1800
png @OUTPUT_PNG@, dpi=300

quit
