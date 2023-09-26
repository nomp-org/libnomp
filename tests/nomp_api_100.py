import loopy as lp

LOOPY_LANG_VERSION = (2018, 2)


def transform(knl, context):
    (iname,) = knl.default_entrypoint.all_inames()
    block_size = min(512, context["device::max_threads_per_block"])
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, block_size, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    return knl


def function_with_syntax_error(knl, context):
    (iname,) = knl.default_entrypoint.all_inames()
    block_size = min(512, context["device::max_threads_per_block"])
    i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
    knl = lp.split_iname(
        knl, iname, block_size, inner_iname=i_inner, outer_iname=i_outer
    )
    knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
    # pylint-disable-reason: This is a test.
    # pylint: disable=E0602
    return kn  # noqa
