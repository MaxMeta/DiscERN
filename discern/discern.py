
import argparse
import pathlib
import pickle
import sys
from .funcs_2 import *
from .vector_tools import *
from .polymer_tools import *
from .low_n_mode import run_low_n_search
from .custom_refs import (CustomReferenceError, cap_min_k, custom_mode_notice,
                          load_custom_references)
from . import cb_sources, homology, mibig_db

#should work with AntiSMASH < 7 if corresponding mibig ref files provided ... todo

_THIS_DIR = pathlib.Path(__file__).parent.resolve()
_DEFAULT_DATA_DIR = _THIS_DIR / "data"
_DEFAULT_DB_DIR = _THIS_DIR / "bigslice_models"

# --- Define Default Paths to Bundled Data ---
_DEFAULT_MIBIG_CB_FILE = _DEFAULT_DATA_DIR / "mibig_cb_dict.json"
_DEFAULT_MIBIG_BS_FILE = _DEFAULT_DATA_DIR / "mibig_bs_vecs.tsv"
_DEFAULT_FEATURE_DICT_FILE = _DEFAULT_DATA_DIR / "feature_dict.json"
_DEFAULT_MIBIG_COUNT_FILE = _DEFAULT_DATA_DIR / "mibig_counts_rf.json"
_DEFAULT_MIBIG_POL_FILE = _DEFAULT_DATA_DIR / "mibig_pol_dict.pkl"

def _discern_version():
    try:
        from importlib.metadata import version
        return version("DiscERN")
    except Exception:
        return None


def _git_commit():
    import subprocess
    try:
        result = subprocess.run(["git", "-C", str(_THIS_DIR), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or None
    except Exception:
        return None


def write_run_info(output_dir, args, custom_dir, mibig_refs, custom_labels, mibig_exclude,
                   mibig_set, settings, tree_min_k, max_k):
    """Record how the run was set up in <output_dir>/run_info.json."""
    if mibig_set is not None:
        mibig = {
            "set": mibig_set.name,
            "releases": mibig_set.manifest["releases"],
            "built": mibig_set.manifest.get("built"),
            "background_diamond_version": mibig_set.manifest.get("diamond_version"),
        }
    else:
        mibig = {"data": "bundled KnownClusterBlast-based files (mibig_cb_dict.json, mibig_bs_vecs.tsv)"}
    info = {
        "discern_version": _discern_version(),
        "git_commit": _git_commit(),
        "command": sys.argv,
        "cb_source": args.cb_source,
        "mibig": mibig,
        "diamond_version": homology.diamond_version() if mibig_set is not None else None,
        "homology_settings": settings.as_dict() if settings is not None else None,
        "references": {
            "mibig": sorted(mibig_refs),
            "mibig_dropped_before_4.0": sorted(set(mibig_refs) & mibig_set.legacy_ids()) if mibig_set else [],
            "custom": list(custom_labels),
            "custom_dir": custom_dir,
        },
        "mibig_exclude": sorted(mibig_exclude) if mibig_exclude else [],
        "poly_search": args.poly_search,
        "min_k": args.min_k,
        "tree_min_k": tree_min_k,
        "max_k": max_k,
    }
    with open(os.path.join(output_dir, "run_info.json"), "w") as handle:
        json.dump(info, handle, indent=4)


def main():
    mibig_cb_file=_DEFAULT_MIBIG_CB_FILE
    mibig_bs_file=_DEFAULT_MIBIG_BS_FILE
    feature_dict_file=_DEFAULT_FEATURE_DICT_FILE
    mibig_count_file=_DEFAULT_MIBIG_COUNT_FILE
    mibig_pol_dict_file=_DEFAULT_MIBIG_POL_FILE
    db_dir=_DEFAULT_DB_DIR

    mibig_cb_file = str(mibig_cb_file)
    mibig_bs_file = str(mibig_bs_file)
    feature_dict_file = str(feature_dict_file)
    mibig_count_file = str(mibig_count_file)
    mibig_pol_dict_file = str(mibig_pol_dict_file)

    parser = argparse.ArgumentParser(description="arguments for main script")
    
    parser.add_argument("-a", "--antismash_dir",
                        required=True,
                        help="Base directory containing antismash outputs."
    )

    parser.add_argument("-o", "--output_dir",
                        required=True,
                        default=None, help="path for output folder to create"
    )

    parser.add_argument("-c", "--num_cpus",
                        type=int,
                        default=0,
                        help="Number of CPUs (default: all available physical cores)."
    )

    parser.add_argument("-r", "--reference_bgcs",
                        default=None,
                        help="reference bgcs defining family to be expanded: MIBiG IDs as a "
                             "space separated string or a file with one ID per line, or a "
                             "directory of custom reference BGCs"
    )

    parser.add_argument("--custom_refs",
                        default=None,
                        help="directory of custom reference BGCs to add to the MIBiG references "
                             "given with -r (mixed references)"
    )

    parser.add_argument("--cb_source",
                        choices=list(cb_sources.CB_SOURCES),
                        default="antismash",
                        help="where the blast-vec and BLAST-rank vectors come from: antiSMASH "
                             "KnownClusterBlast text files (antismash, default; the published "
                             "setup) or DiscERN's own DIAMOND search against MIBiG and the custom "
                             "references (diamond; needs discern-setup-mibig, no "
                             "--cb-knownclusters needed)"
    )

    parser.add_argument("--mibig_set",
                        choices=list(mibig_db.MIBIG_SETS),
                        default=None,
                        help="with --cb_source diamond: MIBiG 4.0 as published (4.0, default), or "
                             "4.0 plus the entries dropped since earlier releases (4.0_legacy; "
                             "see docs/mibig_dropped_entries.md)"
    )

    parser.add_argument("--mibig_db_dir",
                        default=None,
                        help="with --cb_source diamond: where discern-setup-mibig built the "
                             f"MIBiG sets (default: ${mibig_db.DB_ROOT_ENV} or "
                             f"{mibig_db.DEFAULT_DB_ROOT})"
    )

    parser.add_argument("--cb_cap",
                        type=int,
                        default=homology.DEFAULT_SETTINGS.cap,
                        help="with --cb_source diamond: subject BGCs kept per blast vector, "
                             "after MIBiG and custom hits are merged (default: 50, as "
                             "antiSMASH KnownClusterBlast; 0 keeps all)"
    )

    parser.add_argument("-b", "--beta",
                        type=float,
                        default=None,
                        help="use f-beta score instead of mcc, define value of beta here"
    )

    parser.add_argument("-u","--reuse", 
                        default=None,
                        help="path to previous output directory to reuse." 
                          "-a should be set to the input dir that was used to make the output"
    )
    
    parser.add_argument("-s","--resmash", 
                        action='store_true',
                        default=False,
                        help="run antismash analysis on combined gbk outputs"
    )

    parser.add_argument("-x","--mibig_exclude", 
                        default=None,
                        help="file or space seperated string of MiBig BGCs to ignore during analysis"
    )

    parser.add_argument("--bigslice_cutoff", 
                        default=None,
                        help="cutoff to use for inclusion with bigslice vectors"
    )
    
    parser.add_argument("--clusterblast_cutoff", 
                        default=None,
                        help="cutoff to use for inclusion with clusterblast vectors"
    )

    parser.add_argument("--hclust", 
                        default=True, 
                        help="conduct hierachical clustering of reference vectors")

    parser.add_argument("-v","--vec_check", 
                        default=True, 
                        help="check ref collection for outliers, and possible mibig entrires that should be added"
                        "useful for optimising bgc familiy definition"
    )

    parser.add_argument(
        "-p", "--poly_search",
        action='store_true',  
        default=False,        
        help="Run polymer prediction based search for NRPS/PKS containing BGCs. "
             "Recommended for NRPS/PKS BGCs with >= 5 modules."
    )
    parser.add_argument(
        "--as6",
        action='store_true',  
        default=False,        
        help="Use antiSMASH 6 outputs"
    ) 
    
    parser.add_argument("-k", "--min_k",
                        type=int,
                        default=3,
                        help="minimum algorithm support for hclust"
    )
    
    parser.add_argument(
        "-l",
        "--low_n_mode",
        action='store_true',
        default=False,
        help="""Use a special 'top-hit consensus' mode for BGC families with few (e.g., <=3 )
             known members. This mode identifies hits where BOTH the top BLAST hit and
             the top predicted polymer structure hit are in the reference family.
             Requires '--poly_search' to be active. Custom references need
             '--cb_source diamond'."""
    )

    args=parser.parse_args()

    antismash_dir=args.antismash_dir
    output_dir=args.output_dir
    num_cpus=args.num_cpus
    ref_set=args.reference_bgcs
    beta=args.beta
    mibig_exclude=args.mibig_exclude
    reuse=args.reuse
    resmash=args.resmash
    hclust=args.hclust
    vec_check=args.vec_check
    as6=args.as6
    low_n=args.low_n_mode
    cb_source=args.cb_source
    diamond=cb_source == "diamond"

    if not args.reference_bgcs and not args.custom_refs:
        parser.error("give MIBiG references with -r, custom references with --custom_refs "
                     "(or -r <directory>), or both")
    if args.reference_bgcs and os.path.isdir(args.reference_bgcs):
        if args.custom_refs:
            parser.error("-r is a directory; to mix references, give MIBiG IDs with -r "
                         "and the custom reference directory with --custom_refs")
        custom_dir = args.reference_bgcs
        ref_set = None
    else:
        custom_dir = args.custom_refs
    if custom_dir and not os.path.isdir(custom_dir):
        parser.error(f"--custom_refs: '{custom_dir}' is not a directory")
    if args.mibig_set and not diamond:
        parser.error("--mibig_set only applies to --cb_source diamond")
    mibig_set_name = args.mibig_set or mibig_db.DEFAULT_SET

    if low_n:
        if not args.poly_search:
            print("Error: --low_n_mode requires --poly_search to be activated.")
            return # Exit
        if custom_dir and not diamond:
            print("Error: --low_n_mode only works with custom references with --cb_source diamond "
                  "(custom references are never a KnownClusterBlast top hit).")
            return # Exit

    if not 1<=args.min_k<=4:
        print("min_k must be between 1-4. Setting to default value of 3")
        args.min_k=3
        
    if args.clusterblast_cutoff:
        ccb=float(args.clusterblast_cutoff)
    else:
        ccb=None
    if args.bigslice_cutoff:
        cbs=float(args.bigslice_cutoff)
    else:
        cbs=None

    ref_dir=False

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Parse mibig_exclude once up front. The 'self' sentinel is resolved later,
    # after ref_set has been parsed.
    if mibig_exclude:
        if os.path.isfile(mibig_exclude):
            with open(mibig_exclude) as F:
                mibig_exclude = {line.strip() for line in F if line.strip()}
        elif mibig_exclude == 'self':
            pass
        else:
            mibig_exclude = set(mibig_exclude.split())

    # MIBiG references (-r IDs or file)
    if not ref_set:
        mibig_refs=set()
    elif os.path.isfile(ref_set):
        lines=set({})
        with open(ref_set) as F:
            for line in F:
                if line.strip():
                    lines.add(line.strip())
        mibig_refs=lines
    else:
        mibig_refs = set(ref_set.split())

    # custom references (-r <directory> or --custom_refs)
    ref_labels=[]
    ref_gbk_paths=[]
    if custom_dir:
        if not diamond:
            print(custom_mode_notice(args.poly_search))
        try:
            (ref_labels, ref_gbk_paths, ref_bs_vecs,
             ref_cb_vecs, ref_pol_vecs) = load_custom_references(
                custom_dir, db_dir, num_cpus=num_cpus,
                poly_search=args.poly_search, as6=as6,
                mibig_exclude=mibig_exclude, cb_source=cb_source)
        except CustomReferenceError as e:
            sys.exit(f"Error: {e}")
        clash=sorted(mibig_refs & set(ref_labels))
        if clash:
            sys.exit(f"Error: custom reference names are also given as MIBiG references: {', '.join(clash)}")
        ref_dir=True

    if ref_dir and mibig_refs:
        ref_set=sorted(mibig_refs)+ref_labels   # mixed references
    elif ref_dir:
        ref_set=ref_labels
    else:
        ref_set=mibig_refs

    tree_min_k, max_k = cap_min_k(args.min_k, ref_dir, args.poly_search, cb_source)
    if tree_min_k < args.min_k:
        print(f"note: --min_k {args.min_k} is above the maximum support a hit can reach "
              f"in this run (k={max_k}); using --min_k {tree_min_k} for the trees")

    print('loading data...')

    with open(feature_dict_file) as F:
        feature_dict=json.load(F)
        
    with open(mibig_count_file) as F:
        mibig_counts=json.load(F)
    
    mibig_set=None
    if diamond:
        # native homology: MIBiG set built by discern-setup-mibig, and Pfam-vec
        # vectors rebuilt from the same MIBiG release(s)
        try:
            homology.find_diamond()
            mibig_set=mibig_db.load_set(mibig_set_name, args.mibig_db_dir)
        except (mibig_db.MibigSetError, homology.DiamondError) as e:
            sys.exit(f"Error: {e}")
        mibig_ids=mibig_set.ids
        mibig_vecs_bs={key:vec for key,vec in cb_sources.load_mibig_bs_vectors(mibig_set_name).items()
                       if key in mibig_ids}
        mibig_vecs_cb=None   # built by the DIAMOND search below
    else:
        mibig_vecs_bs=tsv_to_sparse_dict(mibig_bs_file)

        mibig_vecs_rf={}
        for key in mibig_vecs_bs:
            bgc=os.path.split(key)[-1].split(".")[0]#use BGC number as key
            mibig_vecs_rf[bgc]=mibig_vecs_bs[key]
        mibig_vecs_bs=mibig_vecs_rf

        with open(mibig_cb_file) as F:
            mibig_vecs_cb = json.load(F)
        mibig_ids=set(mibig_vecs_cb)

    with open(mibig_pol_dict_file,'rb') as F:
        mibig_vecs_pol=pickle.load(F)

    # every MIBiG reference needs a blast-vec and a pfam-vec vector
    unknown=sorted(bgc for bgc in mibig_refs if bgc not in mibig_ids or bgc not in mibig_vecs_bs)
    if unknown:
        message=f"Error: no reference data for these MIBiG references: {' '.join(unknown)}."
        dropped=cb_sources.load_dropped_entries()
        legacy=[bgc for bgc in unknown if bgc in dropped]
        if legacy and not (diamond and mibig_set_name=="4.0_legacy"):
            message+=(f" {' '.join(legacy)} {'was' if len(legacy)==1 else 'were'} dropped from MIBiG "
                      "before 4.0 (see docs/mibig_dropped_entries.md); to use them, run with "
                      "--cb_source diamond --mibig_set 4.0_legacy.")
        elif not diamond:
            message+=(" The bundled KnownClusterBlast-based data covers "
                      f"{len(mibig_ids)} MIBiG BGCs; --cb_source diamond covers all of MIBiG 4.0.")
        sys.exit(message)
    if diamond and args.poly_search:
        no_pol=sorted(bgc for bgc in mibig_refs if bgc not in mibig_vecs_pol)
        if no_pol:
            print(f"note: no polymer data for MIBiG references {' '.join(no_pol)}; "
                  "the polymer method cannot match them")

    if mibig_exclude:
        # Resolve the deferred 'self' sentinel now that ref_set is parsed.
        if mibig_exclude == 'self':
            if mibig_refs:
                mibig_exclude = set(mibig_refs)
            else:
                print("ignoring mibig_exclude=self argument as this is not compatible with external ref files")
                mibig_exclude = set()

    if mibig_exclude:
        if not diamond:
            new_cb={key:mibig_vecs_cb[key] for key in mibig_vecs_cb if not key in mibig_exclude}
            for key in new_cb:
                for bgc in mibig_exclude:
                    if bgc in new_cb[key]:
                        del new_cb[key][bgc]
            mibig_vecs_cb=new_cb
        new_bs={key:mibig_vecs_bs[key] for key in mibig_vecs_bs if not key in mibig_exclude}
        new_pol={key:mibig_vecs_pol[key] for key in mibig_vecs_pol if not key in mibig_exclude}
        mibig_vecs_bs=new_bs
        mibig_vecs_pol=new_pol

    if ref_dir:
        #add the externally defined reference vectors to mibig
        mibig_vecs_bs.update(ref_bs_vecs)
        if not diamond:
            mibig_vecs_cb.update(ref_cb_vecs)
        mibig_vecs_pol.update(ref_pol_vecs)

    if reuse and not low_n:
        print("loading previously constructed big-slice vectors from file")
        with open(os.path.join(reuse,"bs_vecs.json")) as F:
            bs_vecs=json.load(F)

    elif not low_n:       
        print("making big-slice vectors")
        bs_vecs = make_sparse_dict(antismash_dir,db_dir)
        with open(os.path.join(output_dir,"bs_vecs.json"),'w') as F:
            json.dump(bs_vecs,F,indent=4)

    settings=homology.HomologySettings(cap=args.cb_cap) if diamond else None
    write_run_info(output_dir, args, custom_dir, mibig_refs, ref_labels, mibig_exclude,
                   mibig_set, settings, tree_min_k, max_k)

    if diamond:
        try:
            cb_vecs, mibig_vecs_cb = cb_sources.diamond_cb_vectors(
                antismash_dir, mibig_set, output_dir, threads=num_cpus,
                custom_refs=dict(zip(ref_labels, ref_gbk_paths)), reuse_dir=reuse,
                settings=settings, exclude=mibig_exclude or ())
        except (CustomReferenceError, homology.DiamondError) as e:
            sys.exit(f"Error: {e}")
        with open(os.path.join(output_dir,f"cb_vecs_diamond_mibig-{mibig_set_name}.json"),'w') as F:
            json.dump(cb_vecs,F,indent=4)
    elif reuse:
        print("loading previously constructed clusterblast vectors from file")
        with open(os.path.join(reuse,"cb_vecs.json")) as F:
            cb_vecs=json.load(F)
    else:
        print("making clusterblast vectors")   
        cb_vecs = parse_cb_outputs(antismash_dir,num_cpus=num_cpus)
        with open(os.path.join(output_dir,"cb_vecs.json"),'w') as F:
            json.dump(cb_vecs,F,indent=4)
            
    if reuse:
        print("loading previously constructed feature counts from file")
        with open(os.path.join(reuse,"feature_counts.json")) as F:
            all_counts=json.load(F)
    else:
        print("making feature counts")  
        all_counts = make_all_feature_counts(antismash_dir,num_cpus,feature_dict)
        all_counts_rf={}
        for key in all_counts:
            all_counts_rf[key]=flatten_nested_dict(all_counts[key])
        all_counts=all_counts_rf
        with open(os.path.join(output_dir,"feature_counts.json"),'w') as F:
            json.dump(all_counts,F,indent=4)
            
    if reuse:
        print("loading previously constructed polymer dict from file")
        with open(os.path.join(reuse,"poly_dict.pkl"),'rb') as F:
            poly_dict=pickle.load(F)
    else:
        print("making polymer dict")
        poly_dict=make_polymer_dict(antismash_dir,as6=as6)
        with open(os.path.join(output_dir,"poly_dict.pkl"),'wb') as F:
            pickle.dump(poly_dict,F)    



    if ref_dir:
        print('finding conserved features in the reference BGCs '
              '(bundled counts for MIBiG references, custom reference gbk files)')
        conserved_features, no_counts = conserved_features_for_references(
            mibig_refs, mibig_counts, ref_gbk_paths, feature_dict)
        if no_counts:
            print(f"note: no bundled feature counts for MIBiG references {' '.join(no_counts)}; "
                  "they are left out of the conserved-domain filter")
    else:
        print('finding conserved features from mibig vecs')
        no_counts=sorted(bgc for bgc in ref_set if bgc not in mibig_counts)
        if no_counts:
            print(f"note: no bundled feature counts for MIBiG references {' '.join(no_counts)}; "
                  "they are left out of the conserved-domain filter")
        conserved_features=flatten_nested_dict(analyse_mibig_set(
            [bgc for bgc in ref_set if bgc in mibig_counts], mibig_counts))
    print(f'found {len(conserved_features)} conserved features')
    if not conserved_features:
        print("warning: no features are conserved across all reference BGCs, "
              "so the domain filter will pass every hit")
    #print(conserved_features)

    
    if low_n:
        # Run the special low-N mode and then finish.
        run_low_n_search(
            cb_vecs=cb_vecs,
            poly_dict=poly_dict,
            mibig_vecs_pol=mibig_vecs_pol,
            ref_set=set(ref_set),
            output_dir=output_dir,
            cb_keys_are_gbk=diamond
        )
        print("DiscERN Low-N Mode finished.")
        if resmash:
            print("running antismash analysis on low N hits")
            gbk_file=os.path.join(output_dir,"low_n_mode_hits.gbk")
            resmash_hits(gbk_file, output_dir=os.path.join(output_dir,"low_n_smash_output"),num_cpus=num_cpus)
        return
    
    print('finding bs vec hits')
    bs_hits, bs_cutoff = find_hits(bs_vecs,mibig_vecs_bs,ref_set,beta=beta,cutoff=cbs)
       
    print('finding cb vec hits')    
    cb_hits, cb_cutoff = find_hits(cb_vecs,mibig_vecs_cb,ref_set,beta=beta,cutoff=ccb)

    if args.poly_search:
        print('finding polymer matches')    
        pol_matches=get_polymer_matches(poly_dict,mibig_vecs_pol)
 
        stats=get_polymer_stats(mibig_vecs_pol,ref_set)
        pol_threshold, pol_f1, pol_precision, pol_recall, pol_tp,\
        pol_fp, pol_fn = find_best_f1_threshold(stats['TP'],stats['FP'],stringent=True)
        pol_hit_dict=filter_polymer_matches(pol_matches,ref_set,pol_threshold)
    
    
        pol_hits=set([polymer_dict_key_to_gbk_path(key) for key in pol_hit_dict])

        #need to debug this (i think)
    
        #print_match_scores(pol_hit_dict,mibig_vecs_pol,poly_dict)
    
    
        match_stats=get_match_stats(pol_hit_dict,mibig_vecs_pol,poly_dict)
    
        with open(os.path.join(output_dir,'polymer_matches.json'),'w') as F:
            json.dump(match_stats,F,indent=4)

    else:
        pol_hits=set({})#empty set for subsequent hits_by_k formatting
            

    if diamond:
        # every reference is a subject of the search, so BLAST-rank runs for
        # MIBiG, custom and mixed references; self hits are dropped by ID
        print('finding ol hits')
        ol_hits=set(find_hits_intersection(mibig_vecs_cb,cb_vecs,ref_set,drop_self_by_id=True))
    elif not ref_dir:
        print('finding ol hits')
        ol_hits=find_hits_intersection(mibig_vecs_cb,cb_vecs,ref_set)
        ol_hits=set(change_file_paths(ol_hits))
    else:
        ol_hits=set({})#empty set for subsequent hits_by_k

    cb_hits=set(change_file_paths(cb_hits))
    bs_hits=set(bs_hits)

    print('filtering by feature counts')



    hits_by_k=find_elements_in_n_of_k_sets([cb_hits,bs_hits,pol_hits,ol_hits])

    passing_hits_by_k={}
    failing_hits_by_k={}


    for k in hits_by_k:
        passing_hits_by_k[k]=filter_by_domains(hits_by_k[k],all_counts,conserved_features)
        failing_hits_by_k[k]=hits_by_k[k].difference(passing_hits_by_k[k])
    
    print('writing output files')

    for k in passing_hits_by_k:
        output_file=os.path.join(output_dir,f'filtered_hits_k{k}.gbk')
        if len(passing_hits_by_k[k])>0:
            combine_genbank_files(passing_hits_by_k[k],output_file)
            
    for k in passing_hits_by_k:
        output_file=os.path.join(output_dir,f'unfiltered_hits_k{k}.gbk')
        if len(failing_hits_by_k[k])>0:
            combine_genbank_files(failing_hits_by_k[k],output_file)
        

    if resmash:
        to_smash=glob.glob(os.path.join(output_dir,'*_hits_k*.gbk'))
        for gbk_file in to_smash:
            out_name=gbk_file.split(".gbk")[0]
            resmash_hits(gbk_file, output_dir=os.path.join(output_dir,out_name),num_cpus=num_cpus)

    if hclust:

        print(poly_dict.keys())

        vecs_to_use=get_vecs_for_trees(passing_hits_by_k,tree_min_k)
        
        make_trees(output_dir,vecs_to_use,ref_set,
                   bs_vecs,cb_vecs,
                   poly_dict,mibig_vecs_bs,
                   mibig_vecs_cb,mibig_vecs_pol,
                   use_pol=args.poly_search,
                   cb_key=(lambda path: path) if diamond else reformat_path_2
                  )
        

    if vec_check and len(ref_set) < 2:
        # analyse_vector_collections needs two or more references
        print("skipping vec_check: it needs at least two reference BGCs")

    elif vec_check:

        chk_cb=analyse_vector_collections(ref_set,mibig_vecs_cb)
        chk_bs=analyse_vector_collections(ref_set,mibig_vecs_bs)
    
        chk_cb_to_save={'outliers':chk_cb['special_outliers_by_distance'],
                'other_close_mibig_bgcs':chk_cb['close_other_vectors'],
                        'max_ref_dist_from_centroid':chk_cb['special_max_distance'],
                        "classification_threshold":cb_cutoff}
        
        chk_bs_to_save={'outliers':chk_bs['special_outliers_by_distance'],
                'other_close_mibig_bgcs':chk_bs['close_other_vectors'],
                        'max_ref_dist_from_centroid':chk_bs['special_max_distance'],
                        "classification_threshold":bs_cutoff}

        with open(os.path.join(output_dir,'blast_vec_check.json'),'w') as F:
            json.dump(chk_cb_to_save,F,indent=4)

        with open(os.path.join(output_dir,'bigslice_vec_check.json'),'w') as F:
            json.dump(chk_bs_to_save,F,indent=4)

        if ref_dir:
            # MIBiG members of the family stay in the background set and
            # count as negatives when the thresholds are trained.
            close_mibig=sorted({bgc for bgc,_ in chk_cb['close_other_vectors']}|
                               {bgc for bgc,_ in chk_bs['close_other_vectors']})
            if close_mibig:
                print("MIBiG BGCs close to the custom reference centroid "
                      "(other_close_mibig_bgcs in blast_vec_check.json / "
                      f"bigslice_vec_check.json): {' '.join(close_mibig)}")
                print("If any of these belong to the family, they were counted as negatives "
                      "when the thresholds were trained; consider re-running with "
                      "-x \"<their IDs>\" to exclude them from the background set.")

    print("writing hit lists to file...")

    with open(os.path.join(output_dir,'passing_hits_by_k.pkl'),'wb') as F:
        pickle.dump(passing_hits_by_k,F)

    with open(os.path.join(output_dir,'failing_hits_by_k.pkl'),'wb') as F:
        pickle.dump(failing_hits_by_k,F)
    
    with open(os.path.join(output_dir,'bs_hits.json'),'w') as F:
        json.dump(list(bs_hits),F,indent=4)

    with open(os.path.join(output_dir,'cb_hits.json'),'w') as F:
        json.dump(list(cb_hits),F,indent=4)

    with open(os.path.join(output_dir,'ol_hits.json'),'w') as F:
        json.dump(list(ol_hits),F,indent=4)

    if args.poly_search:

        with open(os.path.join(output_dir,'polymer_hits.json'),'w') as F:
            json.dump(list(pol_hits),F,indent=4)
    
        

        #print(chk_cb)
        #print(chk_bs)

        
   #make summary text file
    

if __name__ == "__main__":
    
    print('starting DiscERN')

    
    main()
